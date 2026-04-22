"""
federation/clustering.py

Dynamic Theta-guided clustering for UnifiedEdu.

Given a matrix of uploaded Theta vectors (one per institution), this
module computes pairwise L2 distances, runs Ward's agglomerative
clustering, and selects the cut depth that maximises the mean
silhouette score.

Public API
----------
cluster_thetas(thetas, max_clusters) -> ClusterResult
    Main entry point called by the server each t_update rounds.

pairwise_l2(thetas) -> np.ndarray
    Symmetric distance matrix, shape (m, m).

ward_linkage(dist_matrix) -> np.ndarray
    scipy linkage matrix from a condensed distance matrix.

best_cut(linkage_matrix, dist_matrix, max_k) -> Tuple[np.ndarray, int, float]
    Sweep candidate cuts; return (labels, k*, silhouette*).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Output of one clustering pass.

    Attributes
    ----------
    labels : np.ndarray, shape (m,)
        Cluster assignment for each institution (0-indexed).
    num_clusters : int
        Number of clusters K found.
    silhouette : float
        Mean silhouette score at the chosen cut (-1 if K==1 or K==m).
    clusters : Dict[int, List[int]]
        Mapping cluster_id -> list of institution indices.
    dist_matrix : np.ndarray, shape (m, m)
        Pairwise L2 distance matrix used for this clustering.
    linkage_matrix : np.ndarray
        Raw scipy linkage matrix (for dendrogram visualisation).
    """
    labels:         np.ndarray
    num_clusters:   int
    silhouette:     float
    clusters:       Dict[int, List[int]]
    dist_matrix:    np.ndarray
    linkage_matrix: np.ndarray

    def members(self, cluster_id: int) -> List[int]:
        return self.clusters.get(cluster_id, [])

    def cluster_of(self, institution_idx: int) -> int:
        return int(self.labels[institution_idx])

    def __repr__(self) -> str:
        sizes = {k: len(v) for k, v in self.clusters.items()}
        return (
            f"ClusterResult(K={self.num_clusters}, "
            f"silhouette={self.silhouette:.4f}, "
            f"sizes={sizes})"
        )


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def pairwise_l2(thetas: torch.Tensor) -> np.ndarray:
    """
    Compute symmetric pairwise L2 distance matrix.

    Parameters
    ----------
    thetas : Tensor, shape (m, p)
        One Theta vector per institution.

    Returns
    -------
    np.ndarray, shape (m, m), dtype float64
        D[i,j] = ||theta_i - theta_j||_2
    """
    with torch.no_grad():
        # Use broadcasting; cast to float64 for numerical stability
        t = thetas.double()                          # (m, p)
        diff = t.unsqueeze(0) - t.unsqueeze(1)       # (m, m, p)
        dist = diff.norm(dim=-1)                     # (m, m)
    D = dist.cpu().numpy()
    # Enforce exact symmetry and zero diagonal
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# Linkage
# ---------------------------------------------------------------------------

def ward_linkage(dist_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Ward's agglomerative linkage from a square distance matrix.

    scipy.linkage expects a condensed upper-triangular distance vector
    when method='ward' and metric is provided as a precomputed matrix.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (m, m)

    Returns
    -------
    np.ndarray
        scipy linkage matrix of shape (m-1, 4).
    """
    condensed = squareform(dist_matrix, checks=False)
    return linkage(condensed, method="ward")


# ---------------------------------------------------------------------------
# Cut selection
# ---------------------------------------------------------------------------

def best_cut(
    linkage_matrix: np.ndarray,
    dist_matrix:    np.ndarray,
    max_k:          int,
) -> Tuple[np.ndarray, int, float]:
    """
    Sweep candidate cluster counts [2, max_k] and return the cut that
    maximises the mean silhouette score.

    Special cases:
    - m == 1: trivially one cluster, silhouette undefined.
    - max_k == 1: forced single cluster (e.g. early warm-up).
    - All silhouette scores <= 0: fall back to K=2 (or K=max_k if lower).

    Parameters
    ----------
    linkage_matrix : np.ndarray, shape (m-1, 4)
    dist_matrix    : np.ndarray, shape (m, m)
    max_k          : int
        Maximum number of clusters (= floor(m/2)).

    Returns
    -------
    labels     : np.ndarray, shape (m,), 0-indexed cluster assignments
    best_k     : int
    best_score : float   (-1.0 when undefined)
    """
    m = dist_matrix.shape[0]

    if m == 1 or max_k <= 1:
        return np.zeros(m, dtype=int), 1, -1.0

    best_labels: np.ndarray = np.zeros(m, dtype=int)
    best_k:      int        = 2
    best_score:  float      = -2.0   # sentinel below valid silhouette range

    for k in range(2, min(max_k, m) + 1):
        # fcluster with criterion='maxclust' guarantees exactly k clusters
        raw_labels = fcluster(linkage_matrix, k, criterion="maxclust")
        labels_0   = raw_labels - 1   # make 0-indexed

        # Silhouette requires at least 2 distinct labels and < m labels
        unique = np.unique(labels_0)
        if len(unique) < 2:
            continue

        score = silhouette_score(dist_matrix, labels_0, metric="precomputed")
        if score > best_score:
            best_score  = score
            best_k      = k
            best_labels = labels_0

    if best_score == -2.0:
        # No valid cut found (degenerate case); fall back to single cluster
        return np.zeros(m, dtype=int), 1, -1.0

    return best_labels, best_k, float(best_score)


# ---------------------------------------------------------------------------
# Topology-based static clustering (Section 8 baseline)
# ---------------------------------------------------------------------------

def topology_descriptors(thetas: torch.Tensor) -> np.ndarray:
    """
    Compute simple topology descriptors from Theta vectors for use in the
    static-clustering baseline.  Since Theta encodes graph modulations,
    we use the Theta vectors themselves as feature vectors for the initial
    clustering (equivalent to a one-shot clustering at round 0).

    Returns the same distance matrix as pairwise_l2; kept as a separate
    function to make the baseline pipeline explicit.
    """
    return pairwise_l2(thetas)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def cluster_thetas(
    thetas:       torch.Tensor,
    max_clusters: int,
) -> ClusterResult:
    """
    Full clustering pipeline: distance -> Ward linkage -> best cut.

    Parameters
    ----------
    thetas : Tensor, shape (m, p)
        Stacked Theta vectors, one per institution.
    max_clusters : int
        Hard cap on K; typically floor(m / 2).

    Returns
    -------
    ClusterResult
    """
    m = thetas.shape[0]
    if m == 0:
        raise ValueError("cluster_thetas received an empty Theta matrix.")

    dist_matrix = pairwise_l2(thetas)

    # Degenerate cases: linkage requires >= 2 observations
    if m == 1:
        empty_Z = np.zeros((0, 4), dtype=np.float64)
        return ClusterResult(
            labels=np.zeros(1, dtype=int),
            num_clusters=1,
            silhouette=-1.0,
            clusters={0: [0]},
            dist_matrix=dist_matrix,
            linkage_matrix=empty_Z,
        )

    linkage_matrix = ward_linkage(dist_matrix)
    labels, k, sil = best_cut(linkage_matrix, dist_matrix, max_k=max_clusters)

    clusters: Dict[int, List[int]] = {}
    for inst_idx, cid in enumerate(labels.tolist()):
        clusters.setdefault(int(cid), []).append(inst_idx)

    return ClusterResult(
        labels=labels,
        num_clusters=k,
        silhouette=sil,
        clusters=clusters,
        dist_matrix=dist_matrix,
        linkage_matrix=linkage_matrix,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers (used by server.py)
# ---------------------------------------------------------------------------

def intra_cluster_average(
    thetas:  torch.Tensor,
    result:  ClusterResult,
) -> Dict[int, torch.Tensor]:
    """
    Compute per-cluster mean Theta vector.

    Returns
    -------
    Dict[cluster_id, mean_theta]   where mean_theta has shape (p,)
    """
    cluster_means: Dict[int, torch.Tensor] = {}
    for cid, members in result.clusters.items():
        cluster_means[cid] = thetas[members].mean(dim=0)
    return cluster_means


def inter_cluster_average(
    cluster_means: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """
    Average across cluster centroids (unweighted by cluster size, per spec).

    Returns
    -------
    Tensor, shape (p,)
    """
    stacked = torch.stack(list(cluster_means.values()), dim=0)  # (K, p)
    return stacked.mean(dim=0)
