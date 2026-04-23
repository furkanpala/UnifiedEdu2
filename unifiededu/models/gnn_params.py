"""
gnn_params.py

Two Theta parameterisation strategies for modulating frozen backbone layers:

ThetaVector  (homogeneous federation — same backbone per cluster)
    Per-layer LoRA adapters:  W_mod = W + (alpha/rank) * A @ B
    A: (out_features, rank),  kaiming_uniform init
    B: (rank, in_features),   zeros init  =>  delta_W = 0 at start
    Parameter count: p = sum_l (out_f_l*r + r*in_f_l)  (~300 k for distilgpt2 r=8)

ThetaGNN  (heterogeneous federation — arbitrary backbone architectures)
    A small GNN processes the model-graph (nodes = linear layers, chain edges).
    Node features encode layer type, depth, and normalised size.
    Two GraphSAGE-mean steps produce per-layer embeddings decoded to
    (delta_scale, delta_shift) scalars applied as:
        W_mod = W * (1 + tanh(delta_scale) * scale_clip) + delta_shift * std(W)
    Because only normalised features are used, the *same* GNN can modulate
    any backbone (GPT-2, OPT, LLaMA, Pythia …) and its parameters can be
    federated across heterogeneous clients.
    Parameter count: ~17 k (fixed regardless of backbone size).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _soft_sign(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """SoftSign(x, s) = s * x / (s + |x|), element-wise.  Kept for compat."""
    return s * x / (s + x.abs())


# ---------------------------------------------------------------------------
# Strategy 1: LoRA ThetaVector  (homogeneous federation)
# ---------------------------------------------------------------------------

class ThetaVector(nn.Module):
    """
    Per-layer LoRA adapters for a frozen backbone.

    Parameters
    ----------
    layer_shapes : Dict[str, Tuple[int, int]]
        Stable-ordered map  layer_name -> (out_features, in_features).
        Use the dict returned by get_layer_shapes() in client.py.
    lora_rank  : LoRA rank r.
    lora_alpha : Scaling factor; effective delta scale = alpha / rank.
    """

    def __init__(
        self,
        layer_shapes: Dict[str, Tuple[int, int]],
        lora_rank:    int   = 8,
        lora_alpha:   float = 1.0,
    ) -> None:
        super().__init__()
        self.layer_shapes = dict(layer_shapes)
        self.lora_rank    = lora_rank
        self.lora_alpha   = lora_alpha
        self.lora_scale   = lora_alpha / lora_rank

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()

        for name, (out_f, in_f) in self.layer_shapes.items():
            safe = name.replace(".", "__")
            A = torch.empty(out_f, lora_rank)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.lora_A[safe] = nn.Parameter(A)
            self.lora_B[safe] = nn.Parameter(torch.zeros(lora_rank, in_f))

        self.p = sum(
            out_f * lora_rank + lora_rank * in_f
            for out_f, in_f in self.layer_shapes.values()
        )

    def to_flat(self) -> torch.Tensor:
        """Concatenate all LoRA params in stable layer order for federation."""
        parts = []
        for name in self.layer_shapes:
            safe = name.replace(".", "__")
            parts.append(self.lora_A[safe].flatten())
            parts.append(self.lora_B[safe].flatten())
        return torch.cat(parts)

    @property
    def theta(self) -> torch.Tensor:
        """Alias for to_flat() — backward-compatible property."""
        return self.to_flat()

    def extra_repr(self) -> str:
        return f"n_layers={len(self.layer_shapes)}, rank={self.lora_rank}, p={self.p:,}"


def theta_from_flat(
    flat:         torch.Tensor,
    layer_shapes: Dict[str, Tuple[int, int]],
    lora_rank:    int   = 8,
    lora_alpha:   float = 1.0,
) -> ThetaVector:
    """Reconstruct a ThetaVector from a flat tensor after server aggregation."""
    tv = ThetaVector(layer_shapes, lora_rank, lora_alpha).to(flat.device)
    offset = 0
    with torch.no_grad():
        for name, (out_f, in_f) in tv.layer_shapes.items():
            safe = name.replace(".", "__")
            n_A  = out_f * lora_rank
            n_B  = lora_rank * in_f
            tv.lora_A[safe].copy_(flat[offset : offset + n_A].view(out_f, lora_rank))
            offset += n_A
            tv.lora_B[safe].copy_(flat[offset : offset + n_B].view(lora_rank, in_f))
            offset += n_B
    if offset != flat.numel():
        raise ValueError(
            f"theta_from_flat: consumed {offset} elements but flat has {flat.numel()}"
        )
    return tv


# ---------------------------------------------------------------------------
# Strategy 2: ThetaGNN  (heterogeneous federation — architecture-agnostic)
# ---------------------------------------------------------------------------

_LAYER_TYPES = [
    "embedding",
    "attn_qkv",   # combined QKV (GPT-2 c_attn)
    "attn_q",     # separate Q (LLaMA)
    "attn_k",
    "attn_v",
    "attn_out",   # output projection
    "ffn_gate",   # gated activation (LLaMA gate_proj)
    "ffn_up",     # expansion (c_fc / up_proj)
    "ffn_down",   # contraction (c_proj / down_proj)
    "other",
]
_N_TYPES: int = len(_LAYER_TYPES)
_TYPE_IDX: Dict[str, int] = {t: i for i, t in enumerate(_LAYER_TYPES)}


def _infer_layer_type(name: str) -> int:
    n = name.lower()
    if any(x in n for x in ("embed", "wte", "wpe")):
        return _TYPE_IDX["embedding"]
    if any(x in n for x in ("c_attn", "qkv")):
        return _TYPE_IDX["attn_qkv"]
    if "attn" in n and (".q_" in n or "_q_proj" in n):
        return _TYPE_IDX["attn_q"]
    if "attn" in n and (".k_" in n or "_k_proj" in n):
        return _TYPE_IDX["attn_k"]
    if "attn" in n and (".v_" in n or "_v_proj" in n):
        return _TYPE_IDX["attn_v"]
    if any(x in n for x in ("c_proj", "o_proj", "out_proj")):
        return _TYPE_IDX["attn_out"]
    if "gate_proj" in n:
        return _TYPE_IDX["ffn_gate"]
    if any(x in n for x in ("up_proj", "c_fc", "fc1")):
        return _TYPE_IDX["ffn_up"]
    if any(x in n for x in ("down_proj", "fc2")):
        return _TYPE_IDX["ffn_down"]
    return _TYPE_IDX["other"]


def build_node_features(
    layer_names:  List[str],
    layer_shapes: Dict[str, Tuple[int, int]],
) -> torch.Tensor:
    """
    Build (num_layers, node_feat_dim) node feature matrix.

    Features: one-hot type (_N_TYPES), relative depth, log-norm in_f,
    log-norm out_f, log(out_f/in_f) ratio.  All continuous features are
    normalised to [0, 1] so the GNN generalises across backbones.
    """
    n        = len(layer_names)
    d        = _N_TYPES + 4
    feats    = torch.zeros(n, d)
    max_log  = math.log(65537.0)

    for i, name in enumerate(layer_names):
        out_f, in_f = layer_shapes[name]
        feats[i, _infer_layer_type(name)] = 1.0
        feats[i, _N_TYPES + 0] = i / max(n - 1, 1)
        feats[i, _N_TYPES + 1] = math.log(in_f  + 1) / max_log
        feats[i, _N_TYPES + 2] = math.log(out_f + 1) / max_log
        feats[i, _N_TYPES + 3] = math.log(out_f / max(in_f, 1) + 1) / max_log
    return feats


class ThetaGNN(nn.Module):
    """
    Architecture-agnostic Theta via a graph neural network.

    The GNN processes a chain graph of linear layers (nodes = layers,
    edges = sequential connections).  Two GraphSAGE-mean steps produce
    per-layer embeddings decoded to (delta_scale, delta_shift) applied as:

        W_mod = W * (1 + tanh(delta_scale) * scale_clip)
                  + delta_shift * std(W) * shift_clip

    Because only normalised, architecture-independent node features are used,
    the same GNN can be applied to—and federated across—any decoder-only LM.

    Parameters
    ----------
    hidden_dim : GNN hidden dimension.
    scale_clip : bounds multiplicative modulation to [1-c, 1+c].
    shift_clip : bounds additive modulation in units of weight std.
    """

    def __init__(
        self,
        hidden_dim:  int   = 64,
        scale_clip:  float = 0.3,
        shift_clip:  float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale_clip = scale_clip
        self.shift_clip = shift_clip

        feat_dim = _N_TYPES + 4

        self.node_enc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
        )
        # GraphSAGE-mean: concat [self, neighbour_mean] → hidden
        self.sage1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.sage2 = nn.Linear(2 * hidden_dim, hidden_dim)

        # Decoder: zero-init so modulation starts at identity
        self.decoder = nn.Linear(hidden_dim, 2, bias=True)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        self.p = sum(p.numel() for p in self.parameters())

    def _sage_pass(self, h: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        # Chain-graph neighbourhood: predecessor, self, successor
        h_prev = torch.cat([h[:1], h[:-1]], dim=0)
        h_next = torch.cat([h[1:], h[-1:]], dim=0)
        agg    = (h_prev + h + h_next) / 3.0
        return F.relu(layer(torch.cat([h, agg], dim=-1)))

    def forward(
        self,
        layer_names:  List[str],
        layer_shapes: Dict[str, Tuple[int, int]],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns Dict[layer_name, (delta_scale, delta_shift)] — scalar tensors
        with gradients attached through the GNN parameters.
        """
        feats = build_node_features(layer_names, layer_shapes).to(
            next(self.parameters()).device
        )
        h = self.node_enc(feats)
        h = self._sage_pass(h, self.sage1)
        h = self._sage_pass(h, self.sage2)
        out = self.decoder(h)                          # (n, 2)
        ds  = torch.tanh(out[:, 0]) * self.scale_clip # bounded scale delta
        dsh = torch.tanh(out[:, 1]) * self.shift_clip # bounded shift delta
        return {name: (ds[i], dsh[i]) for i, name in enumerate(layer_names)}

    def to_flat(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])

    @property
    def theta(self) -> torch.Tensor:
        return self.to_flat()

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, p={self.p:,}"


def gnn_theta_from_flat(
    flat:       torch.Tensor,
    hidden_dim: int   = 64,
    scale_clip: float = 0.3,
    shift_clip: float = 0.1,
) -> ThetaGNN:
    """Reconstruct a ThetaGNN from a flat tensor after server aggregation."""
    gnn = ThetaGNN(hidden_dim, scale_clip, shift_clip).to(flat.device)
    offset = 0
    with torch.no_grad():
        for p in gnn.parameters():
            n = p.numel()
            p.copy_(flat[offset : offset + n].view(p.shape))
            offset += n
    return gnn
