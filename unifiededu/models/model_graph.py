"""
model_graph.py

Converts any HuggingFace LLM backbone into a directed acyclic graph (DAG)
G_k = (V_k, E_k) where:
  - Nodes carry bias values + activation encoding + optional LayerNorm params
  - Edges carry weight values
  - Topology follows strict topological order (DAG guaranteed)

Supported architectures:
  - Encoder-only (BERT, DistilBERT, RoBERTa)
  - Decoder-only (LLaMA-2, Qwen, GPT-2)
  - Generic fallback via named_parameters() inspection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation encoding
# ---------------------------------------------------------------------------

class ActivationType(IntEnum):
    LINEAR = 0
    GELU   = 1
    RELU   = 2
    SILU   = 3   # used by LLaMA / SwiGLU gate
    TANH   = 4

_ACT_MAP: Dict[str, ActivationType] = {
    "gelu": ActivationType.GELU,
    "relu": ActivationType.RELU,
    "silu": ActivationType.SILU,
    "swish": ActivationType.SILU,
    "tanh": ActivationType.TANH,
}


def _detect_activation(module: nn.Module) -> ActivationType:
    """Heuristically infer the activation type from an nn.Module."""
    name = type(module).__name__.lower()
    for key, act in _ACT_MAP.items():
        if key in name:
            return act
    return ActivationType.LINEAR


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class NodeFeature:
    """Feature vector stored on a DAG node.

    Fields (all scalars):
      bias        -- neuron bias (or 0.0 for input/virtual nodes)
      activation  -- ActivationType int code
      ln_gamma    -- LayerNorm scale absorbed here (0.0 if absent)
      ln_beta     -- LayerNorm shift absorbed here (0.0 if absent)
    """
    bias:       float = 0.0
    activation: int   = ActivationType.LINEAR
    ln_gamma:   float = 0.0
    ln_beta:    float = 0.0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.bias, float(self.activation), self.ln_gamma, self.ln_beta],
            dtype=torch.float32,
        )


@dataclass
class ModelGraph:
    """
    Immutable DAG representation of an LLM backbone.

    Attributes
    ----------
    node_features : Tensor, shape (N, 4)
        One row per node; columns = [bias, act_code, ln_gamma, ln_beta].
    edge_index : Tensor, shape (2, E)
        [source_row; target_row] in COO format (edges directed src -> tgt).
    edge_features : Tensor, shape (E,)
        Weight of each connection.
    node_groups : Tensor, shape (N,)
        Integer group id used by the GNN parameteriser (k_node groups).
    edge_groups : Tensor, shape (E,)
        Integer group id used by the GNN parameteriser (k_edge groups).
    node_layer_ids : Tensor, shape (N,)
        Topological layer index; input nodes = 0.
    input_node_ids : List[int]
        Indices of input nodes (receive external features).
    output_node_ids : List[int]
        Indices of output nodes (read final hidden states from).
    arch_type : str
        Human-readable architecture tag, e.g. "bert-base-uncased".
    """
    node_features:   torch.Tensor
    edge_index:      torch.Tensor
    edge_features:   torch.Tensor
    node_groups:     torch.Tensor
    edge_groups:     torch.Tensor
    node_layer_ids:  torch.Tensor
    input_node_ids:  List[int]
    output_node_ids: List[int]
    arch_type:       str = "unknown"

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_features.shape[0]

    def __repr__(self) -> str:
        return (
            f"ModelGraph({self.arch_type!r}, "
            f"N={self.num_nodes}, E={self.num_edges}, "
            f"node_groups={int(self.node_groups.max())+1}, "
            f"edge_groups={int(self.edge_groups.max())+1})"
        )


# ---------------------------------------------------------------------------
# Internal builder helpers
# ---------------------------------------------------------------------------

class _GraphBuilder:
    """Stateful accumulator used during model-graph construction."""

    def __init__(self) -> None:
        self._node_feats:    List[NodeFeature]      = []
        self._edge_src:      List[int]              = []
        self._edge_tgt:      List[int]              = []
        self._edge_weights:  List[float]            = []
        self._node_layer:    List[int]              = []
        self._node_group:    List[int]              = []
        self._edge_group:    List[int]              = []
        self.input_nodes:    List[int]              = []
        self.output_nodes:   List[int]              = []

    # -- node management ---------------------------------------------------

    def add_node(
        self,
        feat: NodeFeature,
        layer_id: int,
        group_id: int,
    ) -> int:
        idx = len(self._node_feats)
        self._node_feats.append(feat)
        self._node_layer.append(layer_id)
        self._node_group.append(group_id)
        return idx

    def add_input_node(self, group_id: int = 0) -> int:
        idx = self.add_node(NodeFeature(), layer_id=0, group_id=group_id)
        self.input_nodes.append(idx)
        return idx

    # -- edge management ---------------------------------------------------

    def add_edge(self, src: int, tgt: int, weight: float, group_id: int) -> None:
        self._edge_src.append(src)
        self._edge_tgt.append(tgt)
        self._edge_weights.append(weight)
        self._edge_group.append(group_id)

    # -- bulk MLP layer addition -------------------------------------------

    def add_dense_layer(
        self,
        weight: torch.Tensor,   # (out_features, in_features)
        bias:   Optional[torch.Tensor],
        src_node_ids: List[int],
        activation: ActivationType,
        layer_id: int,
        node_group_id: int,
        edge_group_id: int,
        ln_gamma: Optional[torch.Tensor] = None,
        ln_beta:  Optional[torch.Tensor] = None,
    ) -> List[int]:
        """
        Adds one fully-connected layer to the graph.

        Returns the list of newly created target node indices.
        """
        out_features, in_features = weight.shape
        assert len(src_node_ids) == in_features, (
            f"Source nodes ({len(src_node_ids)}) != weight in_features ({in_features})"
        )

        W = weight.detach().float()
        b = bias.detach().float() if bias is not None else torch.zeros(out_features)
        gamma = ln_gamma.detach().float() if ln_gamma is not None else torch.zeros(out_features)
        beta  = ln_beta.detach().float()  if ln_beta  is not None else torch.zeros(out_features)

        tgt_node_ids: List[int] = []
        for j in range(out_features):
            feat = NodeFeature(
                bias=float(b[j]),
                activation=int(activation),
                ln_gamma=float(gamma[j]),
                ln_beta=float(beta[j]),
            )
            nid = self.add_node(feat, layer_id=layer_id, group_id=node_group_id)
            tgt_node_ids.append(nid)

            for m, src in enumerate(src_node_ids):
                self.add_edge(
                    src=src,
                    tgt=nid,
                    weight=float(W[j, m]),
                    group_id=edge_group_id,
                )

        return tgt_node_ids

    # -- finalisation -------------------------------------------------------

    def build(self, arch_type: str, k_node: int, k_edge: int) -> ModelGraph:
        N = len(self._node_feats)
        E = len(self._edge_src)

        node_features = torch.stack([f.to_tensor() for f in self._node_feats])  # (N,4)
        edge_index = torch.tensor(
            [self._edge_src, self._edge_tgt], dtype=torch.long
        )  # (2, E)
        edge_features = torch.tensor(self._edge_weights, dtype=torch.float32)   # (E,)
        node_layer_ids = torch.tensor(self._node_layer, dtype=torch.long)        # (N,)

        # Clamp group ids to [0, k-1]
        raw_ng = torch.tensor(self._node_group, dtype=torch.long)
        raw_eg = torch.tensor(self._edge_group, dtype=torch.long)
        node_groups = raw_ng % k_node
        edge_groups = raw_eg % k_edge

        return ModelGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_groups=node_groups,
            edge_groups=edge_groups,
            node_layer_ids=node_layer_ids,
            input_node_ids=list(self.input_nodes),
            output_node_ids=list(self.output_nodes),
            arch_type=arch_type,
        )


# ---------------------------------------------------------------------------
# Architecture-specific converters
# ---------------------------------------------------------------------------

def _convert_mlp(
    builder: _GraphBuilder,
    linear_layers: List[Tuple[nn.Linear, ActivationType]],
    src_node_ids: List[int],
    base_layer_id: int,
    node_group_base: int,
    edge_group_base: int,
    ln_params: Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
) -> List[int]:
    """
    Appends an MLP stack to the builder.

    Parameters
    ----------
    linear_layers : list of (nn.Linear, activation)
    src_node_ids  : input node indices
    base_layer_id : topological layer offset for the first new layer
    ln_params     : optional list of (gamma, beta) per linear layer
    """
    current_ids = src_node_ids
    for i, (linear, act) in enumerate(linear_layers):
        gamma = beta = None
        if ln_params and i < len(ln_params):
            gamma, beta = ln_params[i]
        current_ids = builder.add_dense_layer(
            weight=linear.weight,
            bias=linear.bias,
            src_node_ids=current_ids,
            activation=act,
            layer_id=base_layer_id + i,
            node_group_id=node_group_base + i,
            edge_group_id=edge_group_base + i,
            ln_gamma=gamma,
            ln_beta=beta,
        )
    return current_ids


def _convert_attention_block(
    builder: _GraphBuilder,
    attn_module: nn.Module,
    src_node_ids: List[int],
    base_layer_id: int,
    node_group_base: int,
    edge_group_base: int,
    num_heads: int,
    hidden_size: int,
    head_dim: int,
    is_cross_attention: bool = False,
) -> List[int]:
    """
    Converts a multi-head attention block into the DAG.

    Handles two common parameter layouts:
      (a) Fused QKV: single weight matrix of shape (3*hidden, hidden)
          -- DistilBERT, some GPT-2 variants
      (b) Separate Q, K, V:  individual nn.Linear modules
          -- BERT, RoBERTa, LLaMA
    """
    # Attempt to locate Q/K/V/O projections by common attribute names
    q_proj = k_proj = v_proj = o_proj = None
    fused_qkv: Optional[nn.Linear] = None

    for attr in ("q_proj", "query"):
        if hasattr(attn_module, attr):
            q_proj = getattr(attn_module, attr); break

    for attr in ("k_proj", "key"):
        if hasattr(attn_module, attr):
            k_proj = getattr(attn_module, attr); break

    for attr in ("v_proj", "value"):
        if hasattr(attn_module, attr):
            v_proj = getattr(attn_module, attr); break

    for attr in ("o_proj", "out_proj", "dense"):
        if hasattr(attn_module, attr):
            o_proj = getattr(attn_module, attr); break

    # Fused QKV (GPT-2: c_attn; Pythia/GPT-NeoX: query_key_value; DistilBERT: qkv)
    for attr in ("c_attn", "query_key_value", "qkv"):
        if hasattr(attn_module, attr):
            fused_qkv = getattr(attn_module, attr); break

    # --- Case A: fused QKV weight ----------------------------------------
    # nn.Linear weight shape: (3*H, H)  -- chunk dim=0
    # Conv1D  weight shape: (H, 3*H)   -- chunk dim=1, then .t() each chunk
    if fused_qkv is not None and q_proj is None:
        is_conv1d = fused_qkv.__class__.__name__ == "Conv1D"
        if is_conv1d:
            # weight: (in_features, 3*hidden) -> chunk along dim=1
            total_out = fused_qkv.weight.shape[1]
            chunk = total_out // 3
            q_weight = fused_qkv.weight[:, :chunk].t().contiguous()
            k_weight = fused_qkv.weight[:, chunk:2*chunk].t().contiguous()
            v_weight = fused_qkv.weight[:, 2*chunk:].t().contiguous()
        else:
            # weight: (3*hidden, in_features) -> chunk along dim=0
            total_out = fused_qkv.weight.shape[0]
            chunk = total_out // 3
            q_weight = fused_qkv.weight[:chunk]
            k_weight = fused_qkv.weight[chunk:2*chunk]
            v_weight = fused_qkv.weight[2*chunk:]

        q_bias = fused_qkv.bias[:chunk]        if fused_qkv.bias is not None else None
        k_bias = fused_qkv.bias[chunk:2*chunk] if fused_qkv.bias is not None else None
        v_bias = fused_qkv.bias[2*chunk:]      if fused_qkv.bias is not None else None

        class _FakeLinear:
            def __init__(self, w, b):
                self.weight = w
                self.bias = b

        q_proj = _FakeLinear(q_weight, q_bias)
        k_proj = _FakeLinear(k_weight, k_bias)
        v_proj = _FakeLinear(v_weight, v_bias)

    if q_proj is None:
        # Cannot identify projections -- skip and pass-through
        return src_node_ids

    layer_offset = base_layer_id

    # Q, K, V projections (run in parallel; same source, consecutive layer ids)
    q_ids = builder.add_dense_layer(
        weight=q_proj.weight, bias=q_proj.bias,
        src_node_ids=src_node_ids,
        activation=ActivationType.LINEAR,
        layer_id=layer_offset,
        node_group_id=node_group_base,
        edge_group_id=edge_group_base,
    )
    k_ids = builder.add_dense_layer(
        weight=k_proj.weight, bias=k_proj.bias,
        src_node_ids=src_node_ids,
        activation=ActivationType.LINEAR,
        layer_id=layer_offset,
        node_group_id=node_group_base + 1,
        edge_group_id=edge_group_base + 1,
    )
    v_ids = builder.add_dense_layer(
        weight=v_proj.weight, bias=v_proj.bias,
        src_node_ids=src_node_ids,
        activation=ActivationType.LINEAR,
        layer_id=layer_offset,
        node_group_id=node_group_base + 2,
        edge_group_id=edge_group_base + 2,
    )

    # Attention output: in the DAG we concatenate Q+K+V node streams
    # into the output projection. For the graph model we treat the
    # attended representation as V projected through O (simplification
    # that preserves all learnable parameters).
    if o_proj is not None:
        # Output projection maps concatenated heads back to hidden_size.
        # Conv1D weight needs transposing to (out, in) orientation.
        attn_out_ids = builder.add_dense_layer(
            weight=_get_linear_weight(o_proj), bias=o_proj.bias,
            src_node_ids=v_ids,
            activation=ActivationType.LINEAR,
            layer_id=layer_offset + 1,
            node_group_id=node_group_base + 3,
            edge_group_id=edge_group_base + 3,
        )
    else:
        attn_out_ids = v_ids

    return attn_out_ids


def _convert_bert(model: nn.Module, cfg) -> Tuple[_GraphBuilder, str]:
    """Convert a BERT-family encoder to a DAG."""
    builder = _GraphBuilder()
    hidden_size = cfg.hidden_size
    num_heads   = cfg.num_attention_heads
    head_dim    = hidden_size // num_heads
    num_layers  = cfg.num_hidden_layers

    # Node group assignment strategy:
    #   0 : input embedding nodes
    #   1 : query projection nodes
    #   2 : key projection nodes
    #   3 : value projection nodes
    #   4 : attention output nodes
    #   5 : FFN intermediate nodes
    #   6 : FFN output nodes
    #   (repeating pattern per transformer block, wrapped mod k_node)

    # --- Embedding layer ---------------------------------------------------
    emb = model.embeddings.word_embeddings  # (vocab, hidden)
    vocab_size = emb.weight.shape[0]

    # Input: one node per hidden dimension
    src_ids = [builder.add_input_node(group_id=0) for _ in range(hidden_size)]

    # We represent the embedding as a linear layer applied to a one-hot input.
    # Practically for the DAG we embed the weight columns directly:
    # embed layer = hidden_size output nodes each reading from hidden_size inputs.
    # Use the embedding weight transposed: (hidden, vocab) -> treat as linear.
    # For manageability we use the mean embedding vector as a single "token" linear.
    # Full vocabulary -> DAG mapping would be intractable for large vocabs.
    # Instead: emit one aggregate node per hidden dim representing the mean weight.
    emb_weight_mean = emb.weight.mean(dim=0, keepdim=True).expand(hidden_size, hidden_size)
    emb_fake = nn.Linear(hidden_size, hidden_size, bias=False)
    emb_fake.weight = nn.Parameter(emb_weight_mean.detach())

    layer_id = 1
    emb_out_ids = builder.add_dense_layer(
        weight=emb_fake.weight, bias=None,
        src_node_ids=src_ids,
        activation=ActivationType.LINEAR,
        layer_id=layer_id,
        node_group_id=0,
        edge_group_id=0,
    )
    layer_id += 1

    current_ids = emb_out_ids

    # --- Transformer blocks ------------------------------------------------
    # Access encoder layers handling both BertModel and BertForMaskedLM wrappers
    encoder = _get_encoder(model)
    if encoder is None:
        raise ValueError("Cannot locate encoder layers in model.")

    for block_idx, layer in enumerate(encoder):
        attn = _get_attention(layer)
        ffn_intermediate, ffn_output = _get_ffn(layer)
        ln1, ln2 = _get_layer_norms(layer)

        ng_base = 1 + block_idx * 6
        eg_base = 1 + block_idx * 6

        # Self-attention
        pre_attn_ids = current_ids
        attn_ids = _convert_attention_block(
            builder=builder,
            attn_module=attn,
            src_node_ids=current_ids,
            base_layer_id=layer_id,
            node_group_base=ng_base,
            edge_group_base=eg_base,
            num_heads=num_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
        )
        layer_id += 2

        # Residual + LayerNorm 1: absorb LN params into attn_ids nodes
        if ln1 is not None and hasattr(ln1, 'weight'):
            _absorb_layer_norm(builder, attn_ids, ln1)

        # Residual skip: x + Attention(x)
        if attn_ids is not pre_attn_ids and len(attn_ids) == len(pre_attn_ids):
            _add_residual_edges(builder, pre_attn_ids, attn_ids, eg_base)

        current_ids = attn_ids

        # FFN
        if ffn_intermediate is not None and ffn_output is not None:
            pre_ffn_ids = current_ids
            act_type = _infer_bert_ffn_activation(layer)
            ffn_ids = builder.add_dense_layer(
                weight=ffn_intermediate.weight, bias=ffn_intermediate.bias,
                src_node_ids=current_ids,
                activation=act_type,
                layer_id=layer_id,
                node_group_id=ng_base + 4,
                edge_group_id=eg_base + 4,
            )
            layer_id += 1
            ffn_out_ids = builder.add_dense_layer(
                weight=ffn_output.dense.weight if hasattr(ffn_output, 'dense') else ffn_output.weight,
                bias=ffn_output.dense.bias if hasattr(ffn_output, 'dense') else ffn_output.bias,
                src_node_ids=ffn_ids,
                activation=ActivationType.LINEAR,
                layer_id=layer_id,
                node_group_id=ng_base + 5,
                edge_group_id=eg_base + 5,
            )
            layer_id += 1

            if ln2 is not None and hasattr(ln2, 'weight'):
                _absorb_layer_norm(builder, ffn_out_ids, ln2)

            # Residual skip: x + FFN(x)
            if len(ffn_out_ids) == len(pre_ffn_ids):
                _add_residual_edges(builder, pre_ffn_ids, ffn_out_ids, eg_base + 5)

            current_ids = ffn_out_ids

    builder.output_nodes = list(current_ids)
    return builder


def _convert_llama(model: nn.Module, cfg) -> _GraphBuilder:
    """Convert a LLaMA-2 / Qwen decoder-only model to a DAG."""
    builder = _GraphBuilder()
    hidden_size = cfg.hidden_size
    num_heads   = cfg.num_attention_heads
    head_dim    = hidden_size // num_heads

    # Input nodes
    src_ids = [builder.add_input_node(group_id=0) for _ in range(hidden_size)]
    layer_id = 1

    # Embedding
    embed_tokens = _find_submodule(model, ("model.embed_tokens", "embed_tokens", "transformer.wte"))
    if embed_tokens is not None:
        emb_w = embed_tokens.weight
        emb_mean = emb_w.mean(dim=0, keepdim=True).expand(hidden_size, hidden_size)
        emb_fake = nn.Linear(hidden_size, hidden_size, bias=False)
        emb_fake.weight = nn.Parameter(emb_mean.detach())
        emb_out_ids = builder.add_dense_layer(
            weight=emb_fake.weight, bias=None,
            src_node_ids=src_ids,
            activation=ActivationType.LINEAR,
            layer_id=layer_id,
            node_group_id=0,
            edge_group_id=0,
        )
        layer_id += 1
        current_ids = emb_out_ids
    else:
        current_ids = src_ids

    # Transformer decoder layers
    decoder_layers = _get_decoder_layers(model)
    if decoder_layers is None:
        raise ValueError("Cannot locate decoder layers in model.")

    for block_idx, layer in enumerate(decoder_layers):
        ng_base = 1 + block_idx * 7
        eg_base = 1 + block_idx * 7

        # Self-attention (LLaMA uses self_attn)
        attn = _find_submodule_attr(layer, ("self_attn", "attention"))
        if attn is not None:
            pre_attn_ids = current_ids
            attn_ids = _convert_attention_block(
                builder=builder,
                attn_module=attn,
                src_node_ids=current_ids,
                base_layer_id=layer_id,
                node_group_base=ng_base,
                edge_group_base=eg_base,
                num_heads=num_heads,
                hidden_size=hidden_size,
                head_dim=head_dim,
            )
            layer_id += 2

            # RMSNorm (LLaMA uses RMSNorm, no beta)
            rms1 = _find_submodule_attr(layer, ("input_layernorm", "ln_1"))
            if rms1 is not None and hasattr(rms1, 'weight'):
                _absorb_rms_norm(builder, attn_ids, rms1)

            # Residual skip: x + Attention(x)
            if attn_ids is not pre_attn_ids and len(attn_ids) == len(pre_attn_ids):
                _add_residual_edges(builder, pre_attn_ids, attn_ids, eg_base)

            current_ids = attn_ids

        # Gated FFN (LLaMA/Qwen use SwiGLU: gate_proj * up_proj -> down_proj)
        # NOTE: SwiGLU is multiplicative (gate * up), but the DAG is strictly
        # additive.  We approximate by adding cross-edges from up_ids into
        # down_ids.  This DAG is a structural proxy for grouping Theta weights,
        # NOT a mathematically exact forward-pass emulator.
        gate_proj = _find_submodule_attr(layer, ("mlp.gate_proj",))
        up_proj   = _find_submodule_attr(layer, ("mlp.up_proj",))
        down_proj = _find_submodule_attr(layer, ("mlp.down_proj",))

        if gate_proj is not None and up_proj is not None and down_proj is not None:
            pre_ffn_ids = current_ids
            # Gate branch (SiLU activation)
            gate_ids = builder.add_dense_layer(
                weight=gate_proj.weight, bias=gate_proj.bias,
                src_node_ids=current_ids,
                activation=ActivationType.SILU,
                layer_id=layer_id,
                node_group_id=ng_base + 4,
                edge_group_id=eg_base + 4,
            )
            # Up branch (linear)
            up_ids = builder.add_dense_layer(
                weight=up_proj.weight, bias=up_proj.bias,
                src_node_ids=current_ids,
                activation=ActivationType.LINEAR,
                layer_id=layer_id,
                node_group_id=ng_base + 5,
                edge_group_id=eg_base + 5,
            )
            layer_id += 1

            down_ids = builder.add_dense_layer(
                weight=down_proj.weight, bias=down_proj.bias,
                src_node_ids=gate_ids,
                activation=ActivationType.LINEAR,
                layer_id=layer_id,
                node_group_id=ng_base + 6,
                edge_group_id=eg_base + 6,
            )
            # Additive proxy for the multiplicative SwiGLU path
            _add_cross_edges(builder, up_ids, down_ids, eg_base + 5)
            layer_id += 1

            rms2 = _find_submodule_attr(layer, ("post_attention_layernorm", "ln_2"))
            if rms2 is not None and hasattr(rms2, 'weight'):
                _absorb_rms_norm(builder, down_ids, rms2)

            # Residual skip: x + FFN(x)
            if len(down_ids) == len(pre_ffn_ids):
                _add_residual_edges(builder, pre_ffn_ids, down_ids, eg_base + 6)

            current_ids = down_ids

        else:
            # Fallback: standard two-layer FFN (GPT-2 / OPT / Pythia)
            # _get_linear_weight handles Conv1D (in,out) -> transpose to (out,in)
            mlp = _find_submodule_attr(layer, ("mlp", "feed_forward"))
            if mlp is not None:
                fc1 = _find_submodule_attr(mlp, ("fc1", "w1", "dense_h_to_4h", "c_fc"))
                fc2 = _find_submodule_attr(mlp, ("fc2", "w2", "dense_4h_to_h", "c_proj"))
                if fc1 is not None and fc2 is not None:
                    pre_ffn_ids = current_ids
                    ffn_ids = builder.add_dense_layer(
                        weight=_get_linear_weight(fc1), bias=fc1.bias,
                        src_node_ids=current_ids,
                        activation=ActivationType.GELU,
                        layer_id=layer_id,
                        node_group_id=ng_base + 4,
                        edge_group_id=eg_base + 4,
                    )
                    layer_id += 1
                    out_ids = builder.add_dense_layer(
                        weight=_get_linear_weight(fc2), bias=fc2.bias,
                        src_node_ids=ffn_ids,
                        activation=ActivationType.LINEAR,
                        layer_id=layer_id,
                        node_group_id=ng_base + 5,
                        edge_group_id=eg_base + 5,
                    )
                    layer_id += 1

                    # Residual skip: x + FFN(x)
                    if len(out_ids) == len(pre_ffn_ids):
                        _add_residual_edges(builder, pre_ffn_ids, out_ids, eg_base + 5)

                    current_ids = out_ids

    # LM head
    lm_head = _find_submodule(model, ("lm_head",))
    if lm_head is not None:
        lm_ids = builder.add_dense_layer(
            weight=lm_head.weight, bias=lm_head.bias,
            src_node_ids=current_ids,
            activation=ActivationType.LINEAR,
            layer_id=layer_id,
            node_group_id=0,
            edge_group_id=0,
        )
        current_ids = lm_ids

    builder.output_nodes = list(current_ids)
    return builder


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_encoder(model: nn.Module):
    """Return the iterable of transformer encoder layers."""
    for attr in (
        "encoder.layer",        # BERT, RoBERTa
        "transformer.layer",    # DistilBERT
        "bert.encoder.layer",
        "roberta.encoder.layer",
    ):
        obj = _find_submodule(model, (attr,))
        if obj is not None:
            return obj
    return None


def _get_decoder_layers(model: nn.Module):
    """Return the iterable of transformer decoder layers."""
    for attr in (
        "model.layers",         # LLaMA, Qwen
        "transformer.h",        # GPT-2
        "model.decoder.layers", # OPT
        "gpt_neox.layers",      # GPT-NeoX
    ):
        obj = _find_submodule(model, (attr,))
        if obj is not None:
            return obj
    return None


def _get_attention(layer: nn.Module) -> Optional[nn.Module]:
    for attr in ("attention", "self_attn", "attn"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def _get_ffn(layer: nn.Module):
    """Return (intermediate_linear, output_module) for BERT-style FFN."""
    # BERT: layer.intermediate.dense, layer.output.dense
    intermediate = None
    output_mod   = None
    for attr in ("intermediate", "ffn.lin1"):
        if hasattr(layer, attr):
            sub = getattr(layer, attr)
            if hasattr(sub, 'dense'):
                intermediate = sub.dense
            elif isinstance(sub, nn.Linear):
                intermediate = sub
            break
    for attr in ("output", "ffn.lin2"):
        if hasattr(layer, attr):
            sub = getattr(layer, attr)
            if hasattr(sub, 'dense'):
                output_mod = sub
            elif isinstance(sub, nn.Linear):
                output_mod = sub
            break
    return intermediate, output_mod


def _get_layer_norms(layer: nn.Module):
    """Return (ln_after_attn, ln_after_ffn) for a BERT-style layer."""
    ln1 = ln2 = None
    for attr in ("attention.output.LayerNorm", "LayerNorm", "sa_layer_norm"):
        obj = _find_submodule(layer, (attr,))
        if obj is not None:
            ln1 = obj; break
    for attr in ("output.LayerNorm", "output_layer_norm"):
        obj = _find_submodule(layer, (attr,))
        if obj is not None:
            ln2 = obj; break
    return ln1, ln2


def _infer_bert_ffn_activation(layer: nn.Module) -> ActivationType:
    """Detect activation used in BERT FFN (usually GELU)."""
    for attr in ("intermediate", "ffn"):
        if hasattr(layer, attr):
            sub = getattr(layer, attr)
            for sub_attr in ("intermediate_act_fn", "activation"):
                if hasattr(sub, sub_attr):
                    fn = getattr(sub, sub_attr)
                    fn_name = type(fn).__name__.lower() if not callable(fn) else fn.__name__.lower() if hasattr(fn, '__name__') else ""
                    for key, act in _ACT_MAP.items():
                        if key in fn_name:
                            return act
    return ActivationType.GELU  # BERT default


def _absorb_layer_norm(
    builder: _GraphBuilder,
    node_ids: List[int],
    ln: nn.Module,
) -> None:
    """Write LayerNorm gamma/beta into existing node features."""
    if not hasattr(ln, 'weight'):
        return
    gamma = ln.weight.detach().float()
    beta  = ln.bias.detach().float() if ln.bias is not None else torch.zeros_like(gamma)
    for i, nid in enumerate(node_ids):
        if i < len(gamma):
            builder._node_feats[nid].ln_gamma = float(gamma[i])
            builder._node_feats[nid].ln_beta  = float(beta[i])


def _absorb_rms_norm(
    builder: _GraphBuilder,
    node_ids: List[int],
    rms: nn.Module,
) -> None:
    """Write RMSNorm weight (no bias) into ln_gamma channel of existing nodes."""
    if not hasattr(rms, 'weight'):
        return
    gamma = rms.weight.detach().float()
    for i, nid in enumerate(node_ids):
        if i < len(gamma):
            builder._node_feats[nid].ln_gamma = float(gamma[i])


def _add_cross_edges(
    builder: _GraphBuilder,
    src_ids: List[int],
    tgt_ids: List[int],
    edge_group_id: int,
) -> None:
    """Add unit-weight edges from every src to every tgt (residual merge)."""
    for src in src_ids:
        for tgt in tgt_ids:
            builder.add_edge(src=src, tgt=tgt, weight=1.0, group_id=edge_group_id)


def _add_residual_edges(
    builder: _GraphBuilder,
    src_ids: List[int],
    tgt_ids: List[int],
    edge_group_id: int,
) -> None:
    """
    Add identity (weight=1.0) skip-connection edges: src[i] -> tgt[i].

    Only wires pairs with the same index so the DAG stays sparse (N edges,
    not N^2).  Silently skips extra elements if lengths differ.
    """
    for src, tgt in zip(src_ids, tgt_ids):
        builder.add_edge(src=src, tgt=tgt, weight=1.0, group_id=edge_group_id)


def _get_linear_weight(module: nn.Module) -> torch.Tensor:
    """
    Return the weight tensor oriented as (out_features, in_features).

    nn.Linear stores weight as (out, in).
    HuggingFace Conv1D stores weight as (in, out) -- transpose required.
    """
    if module.__class__.__name__ == "Conv1D":
        return module.weight.t()
    return module.weight


def _find_submodule(model: nn.Module, attr_paths: Tuple[str, ...]) -> Optional[nn.Module]:
    """Traverse dotted attribute paths; return first match or None."""
    for path in attr_paths:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def _find_submodule_attr(module: nn.Module, attr_paths: Tuple[str, ...]) -> Optional[nn.Module]:
    return _find_submodule(module, attr_paths)


# ---------------------------------------------------------------------------
# Generic fallback converter
# ---------------------------------------------------------------------------

def _convert_generic(model: nn.Module) -> _GraphBuilder:
    """
    Fallback: walk named_parameters() and convert every nn.Linear found.

    Groups layers by name prefix depth to assign node/edge groups.
    """
    builder = _GraphBuilder()
    hidden_size = 768  # default; will be overridden by first layer

    # Collect all Linear layers in order
    linears: List[Tuple[str, nn.Linear]] = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]

    if not linears:
        raise ValueError("No nn.Linear modules found; cannot build model-graph.")

    in_features = linears[0][1].weight.shape[1]
    src_ids = [builder.add_input_node(group_id=0) for _ in range(in_features)]
    layer_id = 1

    for group_id, (name, linear) in enumerate(linears):
        act = ActivationType.GELU if "intermediate" in name or "fc" in name else ActivationType.LINEAR
        tgt_ids = builder.add_dense_layer(
            weight=linear.weight,
            bias=linear.bias,
            src_node_ids=src_ids,
            activation=act,
            layer_id=layer_id,
            node_group_id=group_id,
            edge_group_id=group_id,
        )
        src_ids = tgt_ids
        layer_id += 1

    builder.output_nodes = list(src_ids)
    return builder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model_graph(
    model: nn.Module,
    k_node: int = 8,
    k_edge: int = 8,
) -> ModelGraph:
    """
    Convert any HuggingFace LLM backbone into a ModelGraph DAG.

    Parameters
    ----------
    model  : nn.Module
        A loaded HuggingFace model (BertModel, LlamaForCausalLM, etc.)
    k_node : int
        Number of node groups for the Theta parameteriser.
    k_edge : int
        Number of edge groups for the Theta parameteriser.

    Returns
    -------
    ModelGraph
    """
    cfg = getattr(model, "config", None)
    arch_type = _detect_arch_type(model, cfg)

    model.eval()
    with torch.no_grad():
        if arch_type.startswith("bert") or arch_type.startswith("roberta") or arch_type.startswith("distilbert"):
            builder = _convert_bert(model, cfg)
        elif any(arch_type.startswith(p) for p in ("llama", "qwen", "gpt2", "gpt_neox", "opt", "mistral", "falcon")):
            builder = _convert_llama(model, cfg)
        else:
            builder = _convert_generic(model)

    graph = builder.build(arch_type=arch_type, k_node=k_node, k_edge=k_edge)
    _validate_dag(graph)
    return graph


def _detect_arch_type(model: nn.Module, cfg) -> str:
    """Return a normalised architecture identifier string."""
    if cfg is not None:
        model_type = getattr(cfg, "model_type", "").lower()
        if model_type:
            name = getattr(cfg, "_name_or_path", model_type)
            return name.lower() if name else model_type

    class_name = type(model).__name__.lower()
    for key in ("llama", "qwen", "bert", "roberta", "distilbert", "gpt2", "gpt_neox", "opt", "mistral", "falcon"):
        if key in class_name:
            return key
    return class_name


def _validate_dag(graph: ModelGraph) -> None:
    """
    Assert that edge_index encodes a valid DAG using topological layer ids.

    Every edge (u -> v) must satisfy layer_id[u] < layer_id[v].
    Raises ValueError on violation.
    """
    if graph.num_edges == 0:
        return
    src = graph.edge_index[0]
    tgt = graph.edge_index[1]
    src_layers = graph.node_layer_ids[src]
    tgt_layers = graph.node_layer_ids[tgt]
    violations = (src_layers >= tgt_layers).sum().item()
    if violations > 0:
        raise ValueError(
            f"DAG validation failed: {violations} edges violate topological ordering "
            f"(src_layer >= tgt_layer). Check _GraphBuilder layer_id assignments."
        )
