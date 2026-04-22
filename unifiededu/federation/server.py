"""
federation/server.py

Central server for the UnifiedEdu federation.

Round lifecycle (called once per federation round t):
  1. server.broadcast()          -> Dict[client_id, flat_theta]
  2. each client calls local_train(theta)  -> uploads updated theta
  3. server.aggregate(round, uploads)  -> updates internal state
     a. re-cluster every t_update rounds (Stage 3)
     b. intra-cluster average every t_ic rounds (Stage 2a)
     c. inter-cluster average every t_bc rounds after T_init (Stage 2b)
  4. repeat from 1

The server never sees raw data -- only flat Theta tensors of shape (p,).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from ..models.gnn_params import ThetaVector, theta_from_flat
from ..config import FederationConfig, ModelGraphConfig
from .clustering import (
    ClusterResult,
    cluster_thetas,
    intra_cluster_average,
    inter_cluster_average,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topology-based feature extraction (for static clustering baseline)
# ---------------------------------------------------------------------------

def _topology_features(graph) -> torch.Tensor:
    """
    Compute a compact topology descriptor vector from a ModelGraph.

    Features (all scalars, concatenated):
      - log(N+1)             node count
      - log(E+1)             edge count
      - log(E/(N+1))         edge density proxy
      - max layer id         depth of the DAG
      - mean out-degree      mean edges per source node
      - std  out-degree
      - edge group entropy   distribution of edges across k_edge groups
    """
    N = graph.num_nodes
    E = graph.num_edges

    if E > 0:
        src = graph.edge_index[0]
        degree = torch.zeros(N)
        degree.scatter_add_(0, src, torch.ones(E))
        mean_deg = degree.float().mean().item()
        std_deg  = degree.float().std().item()
        # edge group distribution entropy
        k_e = int(graph.edge_groups.max().item()) + 1
        counts = torch.zeros(k_e)
        counts.scatter_add_(0, graph.edge_groups, torch.ones(E))
        p = counts / counts.sum()
        entropy = -(p * (p + 1e-9).log()).sum().item()
    else:
        mean_deg = std_deg = entropy = 0.0

    depth = float(graph.node_layer_ids.max().item()) if N > 0 else 0.0

    feat = torch.tensor([
        (N + 1.0) ** 0.5,
        (E + 1.0) ** 0.5,
        float(E) / (N + 1.0),
        depth,
        mean_deg,
        std_deg,
        entropy,
    ], dtype=torch.float32)
    return feat


class FederationServer:
    """
    Aggregation server for UnifiedEdu with dynamic Theta-guided clustering.

    Parameters
    ----------
    num_clients  : int
    fed_config   : FederationConfig
    mg_config    : ModelGraphConfig   (determines k_edge, k_node, and p)
    """

    def __init__(
        self,
        num_clients: int,
        fed_config:  FederationConfig,
        mg_config:   ModelGraphConfig,
    ) -> None:
        self.num_clients = num_clients
        self.cfg         = fed_config
        self.k_edge      = mg_config.k_edge
        self.k_node      = mg_config.k_node
        self.p           = 2 * self.k_edge + 1 + 2 * self.k_node + 1

        # Global Theta -- shared starting point, updated after inter-cluster agg
        self._global_theta: torch.Tensor = ThetaVector(
            self.k_edge, self.k_node
        ).theta.detach().clone()

        # Per-client theta broadcast at the start of each round
        self._client_thetas: Dict[int, torch.Tensor] = {}

        # Current cluster partition (initialised lazily on first upload)
        self._cluster_result: Optional[ClusterResult] = None

        # Sorted list of registered client ids (set on first broadcast)
        self._client_ids: List[int] = list(range(num_clients))

        self._round: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cluster_result(self) -> Optional[ClusterResult]:
        return self._cluster_result

    @property
    def round(self) -> int:
        return self._round

    # ------------------------------------------------------------------
    # Step 1: broadcast
    # ------------------------------------------------------------------

    def broadcast(self) -> Dict[int, torch.Tensor]:
        """
        Return the Theta tensor each client should start its local epoch from.

        Before the first clustering is done, every client receives the same
        global Theta.  After clustering, clients in the same cluster share
        the same cluster-mean Theta (set by the previous aggregate() call).
        """
        return {cid: self._client_thetas.get(cid, self._global_theta).clone()
                for cid in self._client_ids}

    # ------------------------------------------------------------------
    # Step 3: aggregate
    # ------------------------------------------------------------------

    def aggregate(
        self,
        round_num:     int,
        uploads: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Process uploaded Theta vectors and return the theta to seed each
        client with at the start of the NEXT round.

        Parameters
        ----------
        round_num : int   (1-indexed)
        uploads   : Dict[client_id, flat_theta_tensor (p,)]

        Returns
        -------
        Dict[client_id, flat_theta_tensor (p,)]
            Same keys as uploads.
        """
        self._round = round_num
        client_ids  = sorted(uploads.keys())
        self._client_ids = client_ids
        m = len(client_ids)

        # Stack uploaded Thetas into a matrix (m, p)
        theta_matrix = torch.stack([uploads[cid] for cid in client_ids])  # (m, p)

        # ----------------------------------------------------------
        # Stage 3: dynamic re-clustering
        # ----------------------------------------------------------
        max_k = self.cfg.effective_max_clusters(num_clients=self.num_clients)
        if self._cluster_result is None or round_num % self.cfg.t_update == 0:
            self._cluster_result = cluster_thetas(theta_matrix, max_clusters=max_k)
            log.info(
                "Round %d: reclustered -> %s",
                round_num, self._cluster_result,
            )

        result = self._cluster_result

        # ----------------------------------------------------------
        # Stage 2b: inter-cluster aggregation (every t_bc, after T_init)
        # ----------------------------------------------------------
        if round_num >= self.cfg.T_init and round_num % self.cfg.t_bc == 0:
            cluster_means = intra_cluster_average(theta_matrix, result)
            grand_mean    = inter_cluster_average(cluster_means)
            self._global_theta = grand_mean.clone()

            out = {cid: grand_mean.clone() for cid in client_ids}
            log.info(
                "Round %d: inter-cluster aggregation -> broadcasting grand mean",
                round_num,
            )
            self._client_thetas = out
            return out

        # ----------------------------------------------------------
        # Stage 2a: intra-cluster aggregation (every t_ic)
        # ----------------------------------------------------------
        if round_num % self.cfg.t_ic == 0:
            cluster_means = intra_cluster_average(theta_matrix, result)
            out: Dict[int, torch.Tensor] = {}
            for inst_idx, cid in enumerate(client_ids):
                cluster_id = result.cluster_of(inst_idx)
                out[cid]   = cluster_means[cluster_id].clone()
            log.info(
                "Round %d: intra-cluster aggregation (K=%d)",
                round_num, result.num_clusters,
            )
            self._client_thetas = out
            return out

        # ----------------------------------------------------------
        # No aggregation this round -- each client keeps its own theta
        # ----------------------------------------------------------
        out = {cid: uploads[cid].clone() for cid in client_ids}
        self._client_thetas = out
        return out

    # ------------------------------------------------------------------
    # Convenience: run a full federation loop (local simulation)
    # ------------------------------------------------------------------

    def run(
        self,
        clients,     # List[FederationClient]
        num_rounds: Optional[int] = None,
    ) -> None:
        """
        Simulate the full federation loop locally.

        Parameters
        ----------
        clients    : list of FederationClient, indexed by position (client_id = index)
        num_rounds : override cfg.num_rounds if provided
        """
        T = num_rounds if num_rounds is not None else self.cfg.num_rounds

        for t in range(1, T + 1):
            # 1. Broadcast
            thetas_in = self.broadcast()

            # 2. Local training (one epoch per client)
            uploads: Dict[int, torch.Tensor] = {}
            for idx, client in enumerate(clients):
                cid = self._client_ids[idx]
                uploads[cid] = client.local_train(thetas_in[cid])

            # 3. Aggregate
            self.aggregate(t, uploads)

            if t % 10 == 0 or t == 1:
                cr = self._cluster_result
                k  = cr.num_clusters if cr else "?"
                sil = f"{cr.silhouette:.4f}" if cr else "?"
                log.info("Round %d/%d  K=%s  sil=%s", t, T, k, sil)


# ---------------------------------------------------------------------------
# Static-clustering server (Section 8 topology-aware baseline)
# ---------------------------------------------------------------------------

class StaticFederationServer(FederationServer):
    """
    Topology-aware UnifiedEdu with a FIXED partition computed once at init.

    The initial cluster assignment is derived from topology descriptor
    vectors of each client's model-graph (node degrees, edge density, DAG
    depth) — NOT from Theta vectors.  The partition never changes during
    training; only t_ic / t_bc aggregation applies.

    Parameters
    ----------
    graphs       : list of ModelGraph, one per client (order = client_id order)
    num_clients  : int
    fed_config   : FederationConfig
    mg_config    : ModelGraphConfig
    """

    def __init__(
        self,
        graphs,
        num_clients: int,
        fed_config:  FederationConfig,
        mg_config:   ModelGraphConfig,
    ) -> None:
        super().__init__(num_clients, fed_config, mg_config)
        self._fixed_cluster: Optional[ClusterResult] = None
        self._graphs = graphs

    def _init_static_clustering(self) -> None:
        """Cluster once using topology feature vectors."""
        feats  = torch.stack([_topology_features(g) for g in self._graphs])  # (m, F)
        max_k  = self.cfg.effective_max_clusters(num_clients=self.num_clients)
        result = cluster_thetas(feats, max_clusters=max_k)
        self._fixed_cluster = result
        log.info(
            "Static clustering (topology): K=%d  silhouette=%.4f",
            result.num_clusters, result.silhouette,
        )

    def aggregate(
        self,
        round_num:  int,
        uploads:    Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Same aggregation logic but clustering is frozen after round 1."""
        self._round    = round_num
        client_ids     = sorted(uploads.keys())
        self._client_ids = client_ids
        theta_matrix   = torch.stack([uploads[cid] for cid in client_ids])

        # Initialise topology clustering exactly once
        if self._fixed_cluster is None:
            self._init_static_clustering()
        self._cluster_result = self._fixed_cluster   # never updated

        result = self._cluster_result

        # Same two-level aggregation as the dynamic server
        if round_num >= self.cfg.T_init and round_num % self.cfg.t_bc == 0:
            cluster_means = intra_cluster_average(theta_matrix, result)
            grand_mean    = inter_cluster_average(cluster_means)
            self._global_theta = grand_mean.clone()
            out = {cid: grand_mean.clone() for cid in client_ids}
            self._client_thetas = out
            return out

        if round_num % self.cfg.t_ic == 0:
            cluster_means = intra_cluster_average(theta_matrix, result)
            out = {}
            for inst_idx, cid in enumerate(client_ids):
                cluster_id = result.cluster_of(inst_idx)
                out[cid]   = cluster_means[cluster_id].clone()
            self._client_thetas = out
            return out

        out = {cid: uploads[cid].clone() for cid in client_ids}
        self._client_thetas = out
        return out
