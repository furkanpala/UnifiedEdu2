"""
forward_pass.py

DAG forward pass that emulates the LLM computation using the
modulated model-graph.

  H^{(0)}_v = input text features at input nodes, 0 elsewhere
  For each node v in topological order:
    H_v = sigma[ sum_{u: A_{uv}=1} E_{u,v} * H_u + bias_v ]

where sigma is the activation function stored as node_features[:, 1].
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .model_graph import ActivationType, ModelGraph


_ACT_FN = {
    ActivationType.LINEAR: lambda x: x,
    ActivationType.GELU:   F.gelu,
    ActivationType.RELU:   F.relu,
    ActivationType.SILU:   F.silu,
    ActivationType.TANH:   torch.tanh,
}


def dag_forward(
    graph: ModelGraph,
    input_features: torch.Tensor,
) -> torch.Tensor:
    """
    Run a single forward pass through the DAG.

    Parameters
    ----------
    graph : ModelGraph
        A Theta-modulated model graph.
    input_features : Tensor, shape (N_input, feature_dim) or (feature_dim,)
        Feature vectors for input nodes.  If 1-D, broadcast to all inputs.

    Returns
    -------
    Tensor, shape (N_output, feature_dim)
        Hidden states at output nodes.
    """
    N = graph.num_nodes
    device = input_features.device

    # feature_dim: the hidden dimension carried at each node
    if input_features.dim() == 1:
        feature_dim = input_features.shape[0]
        h0 = input_features.unsqueeze(0).expand(len(graph.input_node_ids), -1)
    else:
        feature_dim = input_features.shape[1]
        h0 = input_features

    # Initialise hidden states to zero
    H = torch.zeros(N, feature_dim, device=device, dtype=input_features.dtype)

    # Assign input features
    for i, nid in enumerate(graph.input_node_ids):
        H[nid] = h0[i] if i < h0.shape[0] else torch.zeros(feature_dim, device=device)

    # Topological processing order
    order = _topological_order(graph)

    # Pre-index: for each node, which edges are incoming?
    # Build adjacency as: tgt_to_src[v] = list of (src_idx_in_edge_tensor, src_node_id)
    tgt_to_src = _build_incoming(graph)

    node_feats = graph.node_features    # (N, 4)
    edge_feats = graph.edge_features    # (E,)

    for v in order:
        if v in graph.input_node_ids:
            # Input nodes: features already set; still apply bias and activation
            bias_v = node_feats[v, 0].item()
            act_code = int(node_feats[v, 1].item())
            act_fn = _ACT_FN.get(act_code, lambda x: x)
            H[v] = act_fn(H[v] + bias_v)
        else:
            incoming = tgt_to_src.get(v, [])
            if incoming:
                edge_indices, src_ids = zip(*incoming)
                # Weighted sum: sum_u E_{u,v} * H_u
                weights = edge_feats[list(edge_indices)]          # (in_degree,)
                src_h   = H[list(src_ids)]                        # (in_degree, feature_dim)
                pre_act = (weights.unsqueeze(1) * src_h).sum(0)  # (feature_dim,)
            else:
                pre_act = torch.zeros(feature_dim, device=device)

            bias_v   = node_feats[v, 0].item()
            act_code = int(node_feats[v, 1].item())
            act_fn   = _ACT_FN.get(act_code, lambda x: x)
            H[v]     = act_fn(pre_act + bias_v)

    output_ids = graph.output_node_ids
    return H[output_ids]  # (N_out, feature_dim)


def dag_forward_batched(
    graph: ModelGraph,
    input_features: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorised forward pass -- avoids per-node Python loops by processing
    all edges in one scatter-add per topological layer.

    Parameters
    ----------
    input_features : Tensor, shape (B, N_input, feature_dim)
        Batched input features. B = batch size.

    Returns
    -------
    Tensor, shape (B, N_output, feature_dim)
    """
    B, n_in, feature_dim = input_features.shape
    N = graph.num_nodes
    device = input_features.device
    dtype  = input_features.dtype

    H = torch.zeros(B, N, feature_dim, device=device, dtype=dtype)

    for i, nid in enumerate(graph.input_node_ids):
        if i < n_in:
            H[:, nid, :] = input_features[:, i, :]

    # Group edges by target topological layer for efficient scatter
    layer_ids = graph.node_layer_ids          # (N,)
    max_layer  = int(layer_ids.max().item())

    src_e = graph.edge_index[0]               # (E,)
    tgt_e = graph.edge_index[1]               # (E,)
    w_e   = graph.edge_features               # (E,)
    node_feats = graph.node_features          # (N, 4)

    for l in range(1, max_layer + 1):
        # Nodes in this layer
        layer_mask = (layer_ids == l)                    # (N,) bool
        layer_node_ids = layer_mask.nonzero(as_tuple=True)[0]  # (n_l,)

        # Edges whose target is in this layer
        tgt_in_layer = layer_mask[tgt_e]                 # (E,) bool
        if not tgt_in_layer.any():
            continue

        e_src = src_e[tgt_in_layer]                      # (e_l,)
        e_tgt = tgt_e[tgt_in_layer]                      # (e_l,)
        e_w   = w_e[tgt_in_layer]                        # (e_l,)

        # Weighted messages: (B, e_l, feature_dim)
        msgs = H[:, e_src, :] * e_w.unsqueeze(0).unsqueeze(-1)  # broadcast

        # Scatter-add into targets
        tgt_expand = e_tgt.unsqueeze(0).unsqueeze(-1).expand(B, -1, feature_dim)
        H.scatter_add_(1, tgt_expand, msgs)

        # Add bias + activation for each node in this layer
        biases   = node_feats[layer_node_ids, 0]         # (n_l,)
        act_codes = node_feats[layer_node_ids, 1].long() # (n_l,)

        H[:, layer_node_ids, :] = H[:, layer_node_ids, :] + biases.unsqueeze(0).unsqueeze(-1)

        # Apply activations (per-node, may differ)
        for i, nid in enumerate(layer_node_ids.tolist()):
            act_fn = _ACT_FN.get(int(act_codes[i].item()), lambda x: x)
            H[:, nid, :] = act_fn(H[:, nid, :])

    output_ids = graph.output_node_ids
    return H[:, output_ids, :]  # (B, N_out, feature_dim)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _topological_order(graph: ModelGraph):
    """Return node ids sorted by topological layer id (ascending)."""
    return graph.node_layer_ids.argsort().tolist()


def _build_incoming(graph: ModelGraph) -> Dict[int, list]:
    """
    Build a mapping: node_id -> list of (edge_idx, src_node_id) for incoming edges.
    """
    tgt_to_src: Dict[int, list] = {}
    src_e = graph.edge_index[0].tolist()
    tgt_e = graph.edge_index[1].tolist()
    for edge_idx, (s, t) in enumerate(zip(src_e, tgt_e)):
        tgt_to_src.setdefault(t, []).append((edge_idx, s))
    return tgt_to_src
