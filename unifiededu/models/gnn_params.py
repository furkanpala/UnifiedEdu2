"""
gnn_params.py

Theta vector: the shared GNN parameter vector that modulates all node
and edge features across all institutions' model-graphs.

Theta layout (flat):
  [0 : k_edge]                          Theta_edge        (scale per edge group)
  [k_edge : 2*k_edge]                   Theta_edge_shift  (shift per edge group)
  [2*k_edge]                            Theta_scale_edge  (SoftSign scale, scalar)
  [2*k_edge+1 : 2*k_edge+1+k_node]      Theta_node        (scale per node group)
  [2*k_edge+1+k_node : 2*k_edge+1+2*k_node]  Theta_node_shift
  [2*k_edge+1+2*k_node]                 Theta_scale_node  (SoftSign scale, scalar)

Total length p = 2*k_edge + 1 + 2*k_node + 1
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_graph import ModelGraph


def _theta_size(k_edge: int, k_node: int) -> int:
    return 2 * k_edge + 1 + 2 * k_node + 1


def _soft_sign(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """SoftSign(x, s) = s * x / (s + |x|), element-wise."""
    return s * x / (s + x.abs())


class ThetaVector(nn.Module):
    """
    Learnable Theta parameter vector of fixed length p.

    Parameters
    ----------
    k_edge : int  -- number of edge groups
    k_node : int  -- number of node groups

    The vector is stored as a single flat nn.Parameter so that
    standard optimisers (AdamW) can update it directly.
    """

    def __init__(self, k_edge: int, k_node: int) -> None:
        super().__init__()
        self.k_edge = k_edge
        self.k_node = k_node
        self.p = _theta_size(k_edge, k_node)

        # Initialise following Section 3 spec (conservative init)
        init = torch.zeros(self.p)
        # Theta_edge = 1
        init[:k_edge] = 1.0
        # Theta_edge_shift = 0 (already zero)
        # Theta_scale_edge = 1
        init[2 * k_edge] = 1.0
        # Theta_node = 1
        init[2 * k_edge + 1: 2 * k_edge + 1 + k_node] = 1.0
        # Theta_node_shift = 0 (already zero)
        # Theta_scale_node = 1
        init[-1] = 1.0

        self.theta = nn.Parameter(init)

    # ------------------------------------------------------------------
    # Slice accessors
    # ------------------------------------------------------------------

    @property
    def theta_edge(self) -> torch.Tensor:
        return self.theta[: self.k_edge]

    @property
    def theta_edge_shift(self) -> torch.Tensor:
        return self.theta[self.k_edge: 2 * self.k_edge]

    @property
    def theta_scale_edge(self) -> torch.Tensor:
        return self.theta[2 * self.k_edge].unsqueeze(0)

    @property
    def theta_node(self) -> torch.Tensor:
        s = 2 * self.k_edge + 1
        return self.theta[s: s + self.k_node]

    @property
    def theta_node_shift(self) -> torch.Tensor:
        s = 2 * self.k_edge + 1 + self.k_node
        return self.theta[s: s + self.k_node]

    @property
    def theta_scale_node(self) -> torch.Tensor:
        return self.theta[-1].unsqueeze(0)

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"k_edge={self.k_edge}, k_node={self.k_node}, p={self.p}"


def apply_theta_to_graph(
    graph: ModelGraph,
    theta: ThetaVector,
) -> ModelGraph:
    """
    Return a *new* ModelGraph whose node and edge features have been
    modulated by Theta.  The original graph is not mutated.

    For edge (u->v) in group g:
        E_{u,v} <- SoftSign(
            E_{u,v} * Theta_edge[g] + Theta_edge_shift[g],
            Theta_scale_edge
        )

    For node v in group g:
        V_v[bias] <- SoftSign(
            V_v[bias] * Theta_node[g] + Theta_node_shift[g],
            Theta_scale_node
        )

    LayerNorm params (ln_gamma, ln_beta) and activation codes are
    not modulated by Theta -- they remain as stored.
    """
    # --- Edge modulation ---------------------------------------------------
    eg = graph.edge_groups                    # (E,) long
    scale_e  = theta.theta_edge[eg]           # (E,)
    shift_e  = theta.theta_edge_shift[eg]     # (E,)
    s_e      = theta.theta_scale_edge         # (1,)

    new_edge_feats = _soft_sign(
        graph.edge_features * scale_e + shift_e,
        s_e,
    )

    # --- Node modulation (bias channel only, index 0) ---------------------
    ng = graph.node_groups                    # (N,) long
    scale_n  = theta.theta_node[ng]           # (N,)
    shift_n  = theta.theta_node_shift[ng]     # (N,)
    s_n      = theta.theta_scale_node         # (1,)

    bias_col = graph.node_features[:, 0]      # (N,)
    new_bias = _soft_sign(
        bias_col * scale_n + shift_n,
        s_n,
    )

    new_node_feats = graph.node_features.clone()
    new_node_feats[:, 0] = new_bias

    return ModelGraph(
        node_features=new_node_feats,
        edge_index=graph.edge_index,
        edge_features=new_edge_feats,
        node_groups=graph.node_groups,
        edge_groups=graph.edge_groups,
        node_layer_ids=graph.node_layer_ids,
        input_node_ids=graph.input_node_ids,
        output_node_ids=graph.output_node_ids,
        arch_type=graph.arch_type,
    )


def theta_from_flat(flat: torch.Tensor, k_edge: int, k_node: int) -> ThetaVector:
    """
    Reconstruct a ThetaVector from a flat tensor (used after aggregation).
    """
    p = _theta_size(k_edge, k_node)
    assert flat.shape == (p,), f"Expected flat tensor of size {p}, got {flat.shape}"
    
    # FIX: Move the newly instantiated module to the correct device
    tv = ThetaVector(k_edge, k_node).to(flat.device)
    
    with torch.no_grad():
        tv.theta.copy_(flat)
    return tv