"""
Smoke tests for federation/clustering.py

Run: /c/Users/fp223/AppData/Local/anaconda3/envs/ML/python.exe tests/test_clustering.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from unifiededu.federation.clustering import (
    pairwise_l2,
    ward_linkage,
    best_cut,
    cluster_thetas,
    intra_cluster_average,
    inter_cluster_average,
    ClusterResult,
)


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


def test_pairwise_l2_shape_and_symmetry():
    m, p = 5, 34
    thetas = torch.randn(m, p)
    D = pairwise_l2(thetas)
    _assert(D.shape == (m, m), f"Shape mismatch: {D.shape}")
    _assert(np.allclose(D, D.T), "Not symmetric")
    _assert(np.allclose(np.diag(D), 0), "Diagonal not zero")
    _assert((D >= 0).all(), "Negative distances")


def test_pairwise_l2_known_values():
    # Two vectors differing by 1 in each of 4 dims -> L2 = 2
    t = torch.tensor([[0., 0., 0., 0.], [1., 1., 1., 1.]])
    D = pairwise_l2(t)
    _assert(abs(D[0, 1] - 2.0) < 1e-6, f"Expected 2.0, got {D[0,1]}")


def test_ward_linkage_shape():
    m = 6
    D = np.random.rand(m, m)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    Z = ward_linkage(D)
    _assert(Z.shape == (m - 1, 4), f"Linkage shape: {Z.shape}")


def test_best_cut_two_tight_clusters():
    """Points in two well-separated clusters should give K=2."""
    # Cluster A: 3 points near (0,0), Cluster B: 3 points near (10,0)
    pts = torch.tensor([
        [0.0, 0.1], [0.1, 0.0], [-0.1, 0.05],
        [10.0, 0.1], [9.9, 0.0], [10.1, 0.05],
    ])
    D = pairwise_l2(pts)
    Z = ward_linkage(D)
    labels, k, sil = best_cut(Z, D, max_k=3)
    _assert(k == 2, f"Expected K=2, got {k}")
    _assert(sil > 0.5, f"Expected high silhouette, got {sil:.4f}")
    # Both halves should be in the same cluster
    _assert(labels[0] == labels[1] == labels[2], "Cluster A not cohesive")
    _assert(labels[3] == labels[4] == labels[5], "Cluster B not cohesive")
    _assert(labels[0] != labels[3], "Clusters A and B should differ")


def test_best_cut_single_institution():
    """m=1 must return a single cluster without error."""
    thetas = torch.randn(1, 34)
    result = cluster_thetas(thetas, max_clusters=1)
    _assert(result.num_clusters == 1)
    _assert(result.silhouette == -1.0)


def test_best_cut_max_k_cap():
    """max_k=1 forces a single cluster regardless of data."""
    pts = torch.tensor([[0.0], [10.0], [20.0], [30.0]])
    D = pairwise_l2(pts)
    Z = ward_linkage(D)
    labels, k, sil = best_cut(Z, D, max_k=1)
    _assert(k == 1, f"Expected forced K=1, got {k}")
    _assert((labels == 0).all(), "All should be in cluster 0")


def test_cluster_thetas_output_structure():
    """cluster_thetas returns a valid ClusterResult."""
    m, p = 6, 34
    # Construct two clear groups
    group_a = torch.zeros(3, p)
    group_b = torch.ones(3, p) * 50.0
    thetas = torch.cat([group_a, group_b], dim=0)

    result = cluster_thetas(thetas, max_clusters=3)
    _assert(isinstance(result, ClusterResult))
    _assert(result.num_clusters >= 1)
    _assert(result.labels.shape == (m,))
    _assert(result.dist_matrix.shape == (m, m))
    total_members = sum(len(v) for v in result.clusters.values())
    _assert(total_members == m, f"Member count mismatch: {total_members}")


def test_cluster_thetas_separates_groups():
    """Two tight, well-separated groups should be identified."""
    p = 34
    group_a = torch.randn(3, p) * 0.01            # near origin
    group_b = torch.randn(3, p) * 0.01 + 100.0    # far away
    thetas = torch.cat([group_a, group_b], dim=0)

    result = cluster_thetas(thetas, max_clusters=3)
    _assert(result.num_clusters == 2, f"Expected K=2, got {result.num_clusters}")
    labels = result.labels
    _assert(labels[0] == labels[1] == labels[2], "Group A not cohesive")
    _assert(labels[3] == labels[4] == labels[5], "Group B not cohesive")
    _assert(labels[0] != labels[3], "Groups should differ")


def test_intra_cluster_average():
    """Mean within each cluster should match manual computation."""
    p = 4
    t0 = torch.tensor([1., 0., 0., 0.])
    t1 = torch.tensor([3., 0., 0., 0.])
    t2 = torch.tensor([100., 0., 0., 0.])
    thetas = torch.stack([t0, t1, t2])

    result = ClusterResult(
        labels=np.array([0, 0, 1]),
        num_clusters=2,
        silhouette=0.9,
        clusters={0: [0, 1], 1: [2]},
        dist_matrix=np.zeros((3, 3)),
        linkage_matrix=np.zeros((2, 4)),
    )
    means = intra_cluster_average(thetas, result)
    expected_c0 = torch.tensor([2., 0., 0., 0.])
    _assert(torch.allclose(means[0], expected_c0), f"Cluster 0 mean wrong: {means[0]}")
    _assert(torch.allclose(means[1], t2), f"Cluster 1 mean wrong: {means[1]}")


def test_inter_cluster_average():
    """Grand mean across cluster centroids (unweighted by size)."""
    means = {
        0: torch.tensor([0., 0.]),
        1: torch.tensor([4., 0.]),
    }
    grand = inter_cluster_average(means)
    _assert(torch.allclose(grand, torch.tensor([2., 0.])), f"Got {grand}")


def test_max_clusters_cap_respected():
    """cluster_thetas never returns more clusters than max_clusters."""
    p = 34
    # 8 institutions each with a very distinct Theta
    thetas = torch.eye(8, p) * 1000.0
    result = cluster_thetas(thetas, max_clusters=4)
    _assert(result.num_clusters <= 4, f"Exceeded max_clusters: {result.num_clusters}")


if __name__ == "__main__":
    tests = [
        test_pairwise_l2_shape_and_symmetry,
        test_pairwise_l2_known_values,
        test_ward_linkage_shape,
        test_best_cut_two_tight_clusters,
        test_best_cut_single_institution,
        test_best_cut_max_k_cap,
        test_cluster_thetas_output_structure,
        test_cluster_thetas_separates_groups,
        test_intra_cluster_average,
        test_inter_cluster_average,
        test_max_clusters_cap_respected,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
