"""
data/domain_shift.py

Utilities for detecting and quantifying domain shift between client datasets
and between train / test distributions within a single client.

Three analyses are provided:

  1. centroid_shift(embs_a, embs_b)
       Cosine distance between the L2-normalised centroids of two embedding sets.
       Used to measure how different two corpora are in semantic space.

  2. mmd(embs_a, embs_b, kernel="rbf")
       Maximum Mean Discrepancy with an RBF kernel.  A non-parametric measure of
       distributional shift that does not require class labels.

  3. drift_over_rounds(per_round_thetas)
       Given a sequence of Theta vectors across federation rounds, compute the
       L2 norm of the round-over-round difference.  A sudden spike indicates
       that a client's data distribution changed between rounds (concept drift).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Centroid shift (cosine distance)
# ---------------------------------------------------------------------------

def centroid_shift(
    embs_a: np.ndarray,
    embs_b: np.ndarray,
) -> float:
    """
    Cosine distance between the centroids of two embedding arrays.

    Parameters
    ----------
    embs_a, embs_b : ndarray, shape (N, D)
        L2-normalised embeddings (as produced by preprocessing.embed_texts).

    Returns
    -------
    float in [0, 2]  (0 = identical, 2 = antipodal)
    """
    c_a = embs_a.mean(axis=0)
    c_b = embs_b.mean(axis=0)

    norm_a = np.linalg.norm(c_a) + 1e-9
    norm_b = np.linalg.norm(c_b) + 1e-9

    cos_sim = float(np.dot(c_a / norm_a, c_b / norm_b))
    return float(1.0 - cos_sim)


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (RBF kernel)
# ---------------------------------------------------------------------------

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute the RBF kernel matrix K(X, Y)."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    XX = (X ** 2).sum(axis=1, keepdims=True)
    YY = (Y ** 2).sum(axis=1, keepdims=True)
    sq_dists = XX + YY.T - 2.0 * (X @ Y.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def mmd(
    embs_a:     np.ndarray,
    embs_b:     np.ndarray,
    kernel:     str   = "rbf",
    sigma:      float = 1.0,
    max_samples: int  = 500,
    seed:       int   = 42,
) -> float:
    """
    Unbiased Maximum Mean Discrepancy between two embedding distributions.

    MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Parameters
    ----------
    embs_a, embs_b : ndarray, shape (N, D)
    kernel         : only "rbf" supported
    sigma          : RBF bandwidth (default 1.0)
    max_samples    : subsample each set to at most this many points

    Returns
    -------
    float >= 0  (0 = same distribution)
    """
    rng = np.random.default_rng(seed)

    def _subsample(arr: np.ndarray) -> np.ndarray:
        if len(arr) > max_samples:
            idx = rng.choice(len(arr), max_samples, replace=False)
            return arr[idx]
        return arr

    a = _subsample(embs_a).astype(np.float32)
    b = _subsample(embs_b).astype(np.float32)

    K_aa = _rbf_kernel(a, a, sigma)
    K_bb = _rbf_kernel(b, b, sigma)
    K_ab = _rbf_kernel(a, b, sigma)

    n, m = len(a), len(b)

    # Unbiased estimator: zero out diagonal for within-set terms
    np.fill_diagonal(K_aa, 0.0)
    np.fill_diagonal(K_bb, 0.0)

    term_aa = K_aa.sum() / max(n * (n - 1), 1)
    term_bb = K_bb.sum() / max(m * (m - 1), 1)
    term_ab = K_ab.sum() / max(n * m, 1)

    mmd2 = float(term_aa + term_bb - 2.0 * term_ab)
    return max(0.0, mmd2)


# ---------------------------------------------------------------------------
# Theta drift over federation rounds
# ---------------------------------------------------------------------------

def drift_over_rounds(
    per_round_thetas: Dict[int, np.ndarray],
) -> List[Tuple[int, float]]:
    """
    Compute the L2 norm of the Theta change between consecutive rounds.

    Parameters
    ----------
    per_round_thetas : dict mapping round_number -> flat Theta array (1-D)

    Returns
    -------
    List of (round_number, l2_drift) pairs, starting from round 2.
    """
    rounds = sorted(per_round_thetas.keys())
    drifts: List[Tuple[int, float]] = []

    for i in range(1, len(rounds)):
        r_prev = rounds[i - 1]
        r_curr = rounds[i]
        delta  = per_round_thetas[r_curr] - per_round_thetas[r_prev]
        drifts.append((r_curr, float(np.linalg.norm(delta))))

    return drifts


# ---------------------------------------------------------------------------
# Per-client shift report
# ---------------------------------------------------------------------------

def domain_shift_report(
    client_train_embs: Dict[str, np.ndarray],
    client_test_embs:  Dict[str, np.ndarray],
    sigma:             float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    For every client: compute train->test centroid shift and MMD.
    Also compute inter-client centroid shifts.

    Returns
    -------
    Dict with keys:
        "<client>_train_test" : {"centroid_shift": ..., "mmd": ...}
        "<client_a>_vs_<client_b>" : {"centroid_shift": ...}
    """
    report: Dict[str, Dict[str, float]] = {}

    # Train -> test shift per client
    for name in client_train_embs:
        if name not in client_test_embs:
            continue
        tr = client_train_embs[name]
        te = client_test_embs[name]
        report[f"{name}_train_test"] = {
            "centroid_shift": centroid_shift(tr, te),
            "mmd":            mmd(tr, te, sigma=sigma),
        }

    # Inter-client centroid shift (train vs train)
    names = sorted(client_train_embs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            cs = centroid_shift(client_train_embs[na], client_train_embs[nb])
            report[f"{na}_vs_{nb}"] = {"centroid_shift": cs}

    return report
