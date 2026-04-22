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

import logging
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.optim import AdamW
from tqdm import tqdm

log = logging.getLogger(__name__)

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
        client_id:      int,
        model:          nn.Module,
        dataloader,
        k_edge:         int,
        k_node:         int,
        fed_config:     FederationConfig,
        train_config:   TrainingConfig,
        val_dataloader  = None,
        model_name:     str = "unknown",
        tokenizer       = None,
        sample_context: Optional[str] = None,
        device:         str = "cpu",
    ) -> None:
        self.client_id      = client_id
        self.model          = model.to(device)
        self.dataloader     = dataloader
        self.val_dataloader = val_dataloader
        self.k_edge         = k_edge
        self.k_node         = k_node
        self.fed_config     = fed_config
        self.train_config   = train_config
        self.model_name     = model_name
        self.tokenizer      = tokenizer
        self.sample_context = sample_context
        self.device         = device

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

        accum     = self.train_config.gradient_accumulation_steps
        local_eps = self.train_config.local_epochs
        theta.train()
        optimizer.zero_grad()

        for epoch in range(local_eps):
            pending    = 0
            epoch_loss = 0.0
            n_batches  = 0
            ep_label   = f"C{self.client_id}({self.model_name}) ep{epoch+1}/{local_eps}"
            pbar = tqdm(self.dataloader, desc=ep_label, leave=False, unit="batch")
            for batch in pbar:
                loss = self._compute_loss(batch, theta)
                (loss / accum).backward()
                pending    += 1
                epoch_loss += loss.item()
                n_batches  += 1

                if pending == accum:
                    torch.nn.utils.clip_grad_norm_(theta.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    pending = 0

            # Flush remaining gradients at end of epoch
            if pending > 0:
                torch.nn.utils.clip_grad_norm_(theta.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg = epoch_loss / max(n_batches, 1)
            log.info(
                "C%d(%s) ep%d/%d  avg_loss=%.4f",
                self.client_id, self.model_name, epoch + 1, local_eps, avg,
            )

        return theta.theta.detach().clone()

    def compute_val_loss(self, theta_flat: torch.Tensor) -> float:
        """Mean loss on the validation set; returns nan if no val_dataloader."""
        if self.val_dataloader is None:
            return float("nan")
        theta = theta_from_flat(theta_flat.to(self.device), self.k_edge, self.k_node)
        theta.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                total += self._compute_loss(batch, theta).item()
                n += 1
        return total / max(n, 1)

    def generate_question(
        self,
        theta_flat:     torch.Tensor,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Generate one question from the stored context sample using modulated weights.

        Temporarily swaps modulated weights into the frozen backbone for generation,
        then restores the originals.
        """
        if self.tokenizer is None or not self.sample_context:
            return "[no sample context]"
        try:
            theta     = theta_from_flat(theta_flat.to(self.device), self.k_edge, self.k_node)
            modulated = modulate_params(self.model, theta, self.layer_groups)

            prompt    = f"Context: {self.sample_context}\nQuestion:"
            enc       = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256,
            )
            input_ids = enc["input_ids"].to(self.device)

            named_params = dict(self.model.named_parameters())
            originals: Dict[str, torch.Tensor] = {}
            for pname, val in modulated.items():
                if pname in named_params:
                    originals[pname] = named_params[pname].data.clone()
                    named_params[pname].data.copy_(val)

            try:
                with torch.no_grad():
                    out_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=getattr(self.tokenizer, "pad_token_id", 0) or 0,
                    )
            finally:
                for pname, orig in originals.items():
                    named_params[pname].data.copy_(orig)

            new_ids = out_ids[0][input_ids.shape[1]:]
            return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        except Exception as exc:
            log.debug("generate_question error: %s", exc)
            return "[generation error]"

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
