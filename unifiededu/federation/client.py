"""
federation/client.py

Local training loop for one institution.

The backbone is FULLY FROZEN.  Only Theta is trainable.  Two Theta strategies:

  ThetaVector (default, homogeneous federation)
      Per-layer LoRA: W_mod = W + (alpha/rank) * A @ B
      Gradient: loss -> modulated_params -> A, B

  ThetaGNN (heterogeneous federation, use_gnn_theta=True)
      A small GNN maps layer-graph features to per-layer scale/shift.
      Architecture-agnostic: the same GNN can be federated across
      clients with different backbone models.

torch.func.functional_call injects modulated parameters without mutating
the frozen backbone, making training/generation fully consistent.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.optim import AdamW

log = logging.getLogger(__name__)

from ..models.gnn_params import (
    ThetaVector,
    ThetaGNN,
    theta_from_flat,
    gnn_theta_from_flat,
)
from ..config import FederationConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Layer discovery helpers
# ---------------------------------------------------------------------------

def _is_linear_or_conv1d(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D"


def assign_layer_groups(
    model:  nn.Module,
    k_edge: int,
    k_node: int,
) -> Dict[str, Tuple[int, int]]:
    """
    Map every non-tied linear layer to a (edge_group, node_group) pair.

    Groups are used by the server for clustering; with LoRA they no longer
    control Theta size (every layer gets its own adapter regardless of group).

    Tied weights (e.g. GPT-2 lm_head = transformer.wte) are excluded so
    training and generation are always consistent.
    """
    named_param_keys = {n for n, _ in model.named_parameters()}
    groups: Dict[str, Tuple[int, int]] = {}
    i = 0
    for name, module in model.named_modules():
        if _is_linear_or_conv1d(module) and (name + ".weight") in named_param_keys:
            groups[name] = (i % k_edge, i % k_node)
            i += 1
    return groups


def get_layer_shapes(
    model:        nn.Module,
    layer_groups: Dict[str, Tuple[int, int]],
) -> Dict[str, Tuple[int, int]]:
    """
    Return (out_features, in_features) for every layer in layer_groups.

    Handles nn.Linear (weight: out×in) and HuggingFace Conv1D (weight: in×out).
    Preserves insertion order so the flat representation is stable.
    """
    shapes: Dict[str, Tuple[int, int]] = {}
    modules = dict(model.named_modules())
    for name in layer_groups:
        m = modules[name]
        w = m.weight
        if m.__class__.__name__ == "Conv1D":
            in_f, out_f = w.shape   # Conv1D stores (in, out)
        else:
            out_f, in_f = w.shape   # nn.Linear stores (out, in)
        shapes[name] = (out_f, in_f)
    return shapes


# ---------------------------------------------------------------------------
# Modulation: inject Theta-derived parameters via functional_call
# ---------------------------------------------------------------------------

def modulate_params(
    model:        nn.Module,
    theta,                              # ThetaVector | ThetaGNN
    layer_groups: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    """
    Build a parameter dict for torch.func.functional_call.

    ThetaVector path: W_mod = W + (alpha/rank) * A @ B
    ThetaGNN path:    W_mod = W * (1 + delta_scale) + delta_shift * std(W)

    Biases are passed through unchanged so the optimiser can focus on weights.
    Original weights are detached — gradients flow only through Theta.
    """
    if isinstance(theta, ThetaGNN):
        return _modulate_gnn(model, theta, layer_groups)
    return _modulate_lora(model, theta, layer_groups)


def _modulate_lora(
    model:        nn.Module,
    theta:        ThetaVector,
    layer_groups: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())

    for name in layer_groups:
        m    = modules.get(name)
        safe = name.replace(".", "__")
        if m is None or safe not in theta.lora_A:
            continue

        w     = m.weight.detach()
        A, B  = theta.lora_A[safe], theta.lora_B[safe]
        delta = (A @ B).to(w.dtype)

        if m.__class__.__name__ == "Conv1D":
            # Conv1D weight is (in, out); delta from A@B is (out, in) -> transpose
            delta = delta.T

        params[name + ".weight"] = w + theta.lora_scale * delta

        if m.bias is not None:
            params[name + ".bias"] = m.bias.detach()

    return params


def _modulate_gnn(
    model:        nn.Module,
    theta_gnn:    ThetaGNN,
    layer_groups: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    layer_names   = list(layer_groups.keys())
    layer_shapes  = get_layer_shapes(model, layer_groups)
    modulations   = theta_gnn(layer_names, layer_shapes)

    params: Dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())

    for name in layer_groups:
        m = modules.get(name)
        if m is None or name not in modulations:
            continue
        delta_scale, delta_shift = modulations[name]

        w     = m.weight.detach().float()
        w_std = w.std().clamp(min=1e-6)
        w_mod = w * (1.0 + delta_scale) + delta_shift * w_std
        params[name + ".weight"] = w_mod.to(m.weight.dtype)

        if m.bias is not None:
            params[name + ".bias"] = m.bias.detach()

    return params


# ---------------------------------------------------------------------------
# Functional-call generation (consistent with training)
# ---------------------------------------------------------------------------

def _functional_generate(
    model:          nn.Module,
    tokenizer,
    prompt:         str,
    params:         Dict[str, torch.Tensor],
    max_new_tokens: int,
    device:         str,
) -> str:
    """
    Autoregressive generation via functional_call with KV-cache.

    Applies modulated params the same way as training (no manual weight swap),
    so weight-tied layers are handled consistently across both passes.
    """
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0
    eos_id = getattr(tokenizer, "eos_token_id", None) or pad_id

    enc     = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    cur_ids = enc["input_ids"].to(device)
    all_ids = cur_ids[0].tolist()
    generated: list = []
    past_kv = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            kw = {"input_ids": cur_ids, "use_cache": True}
            if past_kv is not None:
                kw["past_key_values"] = past_kv

            out     = functional_call(model, params, args=(), kwargs=kw, strict=False)
            logits  = out.logits[:, -1, :].float()
            past_kv = getattr(out, "past_key_values", None)

            # Repetition penalty
            for tid in set(all_ids):
                if logits[0, tid] > 0:
                    logits[0, tid] /= 1.3
                else:
                    logits[0, tid] *= 1.3

            # No-repeat trigram
            if len(all_ids) >= 2:
                pfx = tuple(all_ids[-2:])
                for i in range(len(all_ids) - 2):
                    if tuple(all_ids[i : i + 2]) == pfx:
                        logits[0, all_ids[i + 2]] = float("-inf")

            probs = torch.softmax(logits / 0.7, dim=-1)
            nxt   = torch.multinomial(probs, 1).item()
            if nxt == eos_id:
                break
            generated.append(nxt)
            all_ids.append(nxt)
            cur_ids = torch.tensor([[nxt]], device=device)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FederationClient:
    """
    One participating institution in the UnifiedEdu federation.

    Parameters
    ----------
    client_id, model, dataloader, fed_config, train_config : as before.
    k_edge, k_node : group counts for server clustering (not Theta size).
    lora_rank      : LoRA rank when use_gnn_theta=False (default).
    lora_alpha     : LoRA alpha scaling.
    use_gnn_theta  : If True, use ThetaGNN instead of ThetaVector.
                     ThetaGNN is architecture-agnostic and enables cross-
                     architecture federation without cluster restrictions.
    gnn_hidden_dim : GNN hidden dim when use_gnn_theta=True.
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
        model_name:     str   = "unknown",
        tokenizer             = None,
        sample_context: Optional[str] = None,
        device:         str   = "cpu",
        lora_rank:      int   = 8,
        lora_alpha:     float = 1.0,
        use_gnn_theta:  bool  = False,
        gnn_hidden_dim: int   = 64,
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
        self.lora_rank      = lora_rank
        self.lora_alpha     = lora_alpha
        self.use_gnn_theta  = use_gnn_theta
        self.gnn_hidden_dim = gnn_hidden_dim

        # Freeze ALL backbone parameters
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        # Stable layer assignment (computed once from the frozen backbone)
        self.layer_groups = assign_layer_groups(model, k_edge, k_node)
        self.layer_shapes = get_layer_shapes(model, self.layer_groups)

        # Cache theta size
        _tmpl = self.make_theta()
        self.p = _tmpl.p
        log.info(
            "C%d(%s) — %d modulated layers, p=%d (%s)",
            client_id, model_name, len(self.layer_groups), self.p,
            "ThetaGNN" if use_gnn_theta else f"LoRA rank={lora_rank}",
        )

    # ------------------------------------------------------------------
    # Theta factory helpers
    # ------------------------------------------------------------------

    def make_theta(self):
        """Create a fresh, properly initialised Theta on self.device."""
        if self.use_gnn_theta:
            return ThetaGNN(self.gnn_hidden_dim).to(self.device)
        return ThetaVector(self.layer_shapes, self.lora_rank, self.lora_alpha).to(self.device)

    def _theta_from_flat(self, flat: torch.Tensor):
        """Reconstruct Theta from a flat vector (after server aggregation)."""
        if self.use_gnn_theta:
            return gnn_theta_from_flat(flat, self.gnn_hidden_dim)
        return theta_from_flat(flat, self.layer_shapes, self.lora_rank, self.lora_alpha)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def local_train(self, theta_flat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Perform one local epoch of AdamW optimisation.

        Parameters
        ----------
        theta_flat : flat Theta from the server, or None on round 1
            (None triggers a fresh kaiming/zero initialisation).

        Returns
        -------
        Updated flat Theta after one local epoch, shape (p,).
        """
        if theta_flat is None:
            theta = self.make_theta()
        else:
            theta = self._theta_from_flat(theta_flat.clone().to(self.device))

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

            for batch in self.dataloader:
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

    def compute_val_loss(self, theta_flat: Optional[torch.Tensor]) -> float:
        """Mean loss on the validation set; returns nan if no val_dataloader."""
        if self.val_dataloader is None:
            return float("nan")
        if theta_flat is None:
            theta = self.make_theta()
        else:
            theta = self._theta_from_flat(theta_flat.to(self.device))
        theta.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                total += self._compute_loss(batch, theta).item()
                n += 1
        return total / max(n, 1)

    def generate_qa(
        self,
        theta_flat:     Optional[torch.Tensor],
        max_new_tokens: int = 40,
    ) -> str:
        """
        Generate one QA pair from the stored context sample.

        Uses functional_call so modulated params are applied identically to
        training — weight-tied layers handled correctly.
        """
        if self.tokenizer is None or not self.sample_context:
            return "[no sample context]"
        try:
            if theta_flat is None:
                theta = self.make_theta()
            else:
                theta = self._theta_from_flat(theta_flat.to(self.device))
            theta.eval()
            modulated = modulate_params(self.model, theta, self.layer_groups)

            q_prompt = f"Context: {self.sample_context}\n\nQuestion: "
            question = _functional_generate(
                self.model, self.tokenizer, q_prompt, modulated, max_new_tokens, self.device
            )
            a_prompt = f"Context: {self.sample_context}\n\nQuestion: {question}\nAnswer: "
            answer   = _functional_generate(
                self.model, self.tokenizer, a_prompt, modulated, max_new_tokens, self.device
            )
            return f"Q: {question} | A: {answer}"

        except Exception as exc:
            log.debug("generate_qa error: %s", exc)
            return "[generation error]"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: dict, theta) -> torch.Tensor:
        """Forward pass with Theta-modulated backbone; return scalar loss."""
        batch     = {k: v.to(self.device) for k, v in batch.items()}
        modulated = modulate_params(self.model, theta, self.layer_groups)

        outputs = functional_call(
            self.model, modulated, args=(), kwargs=batch, strict=False,
        )

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        labels = batch["labels"]
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
