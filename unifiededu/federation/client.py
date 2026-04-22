"""
federation/client.py

Local training loop for one institution.

The backbone model is FULLY FROZEN.  The only trainable variable is the
Theta vector (ThetaVector).  Theta modulates every linear layer's weights
and biases via SoftSign before each forward pass, so the gradient flows:

  loss  ->  modulated_params  ->  SoftSign  ->  Theta

torch.func.functional_call is used to inject modulated parameters into
the frozen HuggingFace backbone without mutating it.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.optim import AdamW

from ..models.gnn_params import ThetaVector, theta_from_flat, _soft_sign
from ..config import FederationConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Layer-group assignment
# ---------------------------------------------------------------------------


def _is_linear_or_conv1d(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    # Catch Hugging Face's custom GPT-2 Conv1D
    if module.__class__.__name__ == "Conv1D":
        return True
    return False

def assign_layer_groups(
    model:  nn.Module,
    k_edge: int,
    k_node: int,
) -> Dict[str, Tuple[int, int]]:
    """
    Map every nn.Linear in the model to a deterministic (edge_group, node_group).

    Assignment uses the enumeration order of named_modules(), which is stable
    for a given model class, so the same groups are used every time the
    function is called on the same architecture.

    Returns
    -------
    Dict[module_name, (edge_group_id, node_group_id)]
    """
    groups: Dict[str, Tuple[int, int]] = {}
    i = 0
    for name, module in model.named_modules():
        if _is_linear_or_conv1d(module):
            groups[name] = (i % k_edge, i % k_node)
            i += 1
    return groups


# ---------------------------------------------------------------------------
# Theta application to model weights  (module-level, used by both client
# and forward_pass emulation)
# ---------------------------------------------------------------------------

def modulate_params(
    model:        nn.Module,
    theta:        ThetaVector,
    layer_groups: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    """
    Build a dict of Theta-modulated weight/bias tensors for every Linear layer.

    Edge modulation (weights):
        w_mod = SoftSign(w * theta_edge[g] + theta_edge_shift[g], theta_scale_edge)

    Node modulation (biases):
        b_mod = SoftSign(b * theta_node[g] + theta_node_shift[g], theta_scale_node)

    The backbone's original weights are detached (no grad flows to them).
    Gradient flows only through Theta.

    Returns
    -------
    Dict suitable for torch.func.functional_call(..., strict=False).
    Keys match the format produced by model.named_parameters().
    """
    params: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not _is_linear_or_conv1d(module):
            continue
        group_info = layer_groups.get(name)
        if group_info is None:
            continue
        eg, ng = group_info

        # Weight (edge modulation) -- scalar broadcast over full matrix
        w = module.weight.detach()
        w_mod = _soft_sign(
            w.float() * theta.theta_edge[eg] + theta.theta_edge_shift[eg],
            theta.theta_scale_edge,
        )
        params[name + ".weight"] = w_mod.to(w.dtype)

        # Bias (node modulation)
        if module.bias is not None:
            b = module.bias.detach()
            b_mod = _soft_sign(
                b.float() * theta.theta_node[ng] + theta.theta_node_shift[ng],
                theta.theta_scale_node,
            )
            params[name + ".bias"] = b_mod.to(b.dtype)

    return params


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FederationClient:
    """
    One participating institution in the UnifiedEdu federation.

    Parameters
    ----------
    client_id   : int
    model       : nn.Module
        HuggingFace backbone (BertForMaskedLM, LlamaForCausalLM, etc.).
        Must support forward(**batch) -> loss when batch contains 'labels'.
    dataloader  : DataLoader
        Yields dicts with keys 'input_ids', 'attention_mask', 'labels'.
    k_edge, k_node : int
        Must match the server's ThetaVector dimensions.
    fed_config  : FederationConfig
    train_config: TrainingConfig
    device      : str
    """

    def __init__(
        self,
        client_id:    int,
        model:        nn.Module,
        dataloader,
        k_edge:       int,
        k_node:       int,
        fed_config:   FederationConfig,
        train_config: TrainingConfig,
        device:       str = "cpu",
    ) -> None:
        self.client_id    = client_id
        self.model        = model.to(device)
        self.dataloader   = dataloader
        self.k_edge       = k_edge
        self.k_node       = k_node
        self.fed_config   = fed_config
        self.train_config = train_config
        self.device       = device

        # Freeze ALL backbone parameters
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        # Stable layer-group assignment (computed once)
        self.layer_groups = assign_layer_groups(model, k_edge, k_node)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def local_train(self, theta_flat: torch.Tensor) -> torch.Tensor:
        """
        Perform one local epoch of mini-batch AdamW optimisation.

        Parameters
        ----------
        theta_flat : Tensor, shape (p,)
            Theta broadcast from the server this round.

        Returns
        -------
        Tensor, shape (p,)
            Updated Theta after one local epoch.
        """
        theta = theta_from_flat(theta_flat.clone().to(self.device), self.k_edge, self.k_node)

        optimizer = AdamW(
            theta.parameters(),
            lr=self.train_config.lr,
            betas=(self.train_config.beta1, self.train_config.beta2),
            weight_decay=self.train_config.weight_decay,
        )

        accum   = self.train_config.gradient_accumulation_steps
        theta.train()
        optimizer.zero_grad()
        pending = 0

        for step, batch in enumerate(self.dataloader):
            loss = self._compute_loss(batch, theta)
            (loss / accum).backward()
            pending += 1

            if pending == accum:
                torch.nn.utils.clip_grad_norm_(theta.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                pending = 0

        # Flush any remaining accumulated gradients
        if pending > 0:
            torch.nn.utils.clip_grad_norm_(theta.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        return theta.theta.detach().clone()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: dict, theta: ThetaVector) -> torch.Tensor:
        """
        Forward pass with Theta-modulated backbone weights; return scalar loss.

        Expects batch keys: 'input_ids', 'attention_mask', 'labels'.
        Labels should be -100 for context (masked) positions.
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        modulated = modulate_params(self.model, theta, self.layer_groups)

        outputs = functional_call(
            self.model,
            modulated,
            args=(),
            kwargs=batch,
            strict=False,
        )

        # HuggingFace models return an object with .loss when labels are provided
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        # Fallback: compute cross-entropy manually from logits
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        labels = batch["labels"]
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
