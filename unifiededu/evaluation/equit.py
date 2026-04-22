"""
evaluation/equity.py

Equity metrics:
  - Curricular Richness C_k
  - Pairwise Equity Score E_{kl}^{(m)}
  - Knowledge Transfer Index KTI(l <- k)
  - Anchor Question Equity Score
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

EPSILON = 1e-6


def curricular_richness(
    samples:     List[dict],
    max_pages:   int,
    alpha:       Tuple[float, float, float] = (0.33, 0.34, 0.33),
) -> float:
    """
    Compute curricular richness score C_k in [0, 1].

    C_k = alpha_1 * (|D_k| / |D_max|)
        + alpha_2 * Diversity(D_k)
        + alpha_3 * Quality(D_k)

    Parameters
    ----------
    samples   : list of sample dicts (must have 'embedding' and 'metadata' keys)
    max_pages : total pages in the largest dataset (for normalisation)
    alpha     : weights (alpha_1, alpha_2, alpha_3), must sum to 1

    Returns
    -------
    float in [0, 1]
    """
    alpha_1, alpha_2, alpha_3 = alpha

    # Component 1: size
    n_pages = len(samples)
    size_score = min(1.0, n_pages / max(1, max_pages))

    # Component 2: diversity (avg pairwise cosine distance of embeddings)
    embs = [s.get("embedding", []) for s in samples if s.get("embedding")]
    if len(embs) >= 2:
        embs_arr = np.array(embs, dtype=np.float32)
        # Normalise
        norms = np.linalg.norm(embs_arr, axis=1, keepdims=True) + 1e-9
        embs_norm = embs_arr / norms
        # Sample up to 500 for efficiency
        idx = np.random.choice(len(embs_norm), min(500, len(embs_norm)), replace=False)
        sub = embs_norm[idx]
        sim_matrix = sub @ sub.T
        # Cosine distance = 1 - cosine similarity
        n = len(sub)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        diversity_score = float((1 - sim_matrix[mask]).mean())
    else:
        diversity_score = 0.0

    # Component 3: quality proxy (source type)
    source_scores = {
        "paper": 1.0, "publication": 1.0,
        "textbook": 0.7,
        "lecture": 0.4, "slides": 0.4, "notes": 0.4,
        "assignment": 0.3,
    }
    quality_vals = []
    for s in samples:
        src = str(s.get("metadata", {}).get("source_type", "")).lower()
        score = 0.4  # default
        for key, val in source_scores.items():
            if key in src:
                score = val
                break
        quality_vals.append(score)
    quality_score = float(np.mean(quality_vals)) if quality_vals else 0.4

    return (
        alpha_1 * size_score
        + alpha_2 * diversity_score
        + alpha_3 * quality_score
    )


def compute_pairwise_equity(
    quality_k: float,
    quality_l: float,
    richness_k: float,
    richness_l: float,
    resource_k: float,
    resource_l: float,
    epsilon:    float = EPSILON,
) -> float:
    """
    Pairwise Equity Score E_{kl}^{(m)}.

    E_{kl} = (Q_k + Q_l) / (|C_k - C_l| + |R_k - R_l| + epsilon)

    Returns
    -------
    float (higher = more equitable)
    """
    numerator   = quality_k + quality_l
    denominator = abs(richness_k - richness_l) + abs(resource_k - resource_l) + epsilon
    return numerator / denominator


def compute_kti(
    quality_uni_on_exclusive:   float,
    quality_indiv_on_exclusive: float,
) -> float:
    """
    Knowledge Transfer Index KTI(l <- k).

    KTI = Q_l^{uni}(T_{k\\l}) - Q_l^{indiv}(T_{k\\l})

    A positive value indicates genuine knowledge transfer from k to l
    on topics exclusive to k's curriculum.

    Parameters
    ----------
    quality_uni_on_exclusive   : Q_l^{uni} evaluated on T_{k\\l}
    quality_indiv_on_exclusive : Q_l^{indiv} evaluated on T_{k\\l}

    Returns
    -------
    float (positive = knowledge transferred)
    """
    return quality_uni_on_exclusive - quality_indiv_on_exclusive


def compute_anchor_equity(
    score_k_indiv: float,
    score_l_indiv: float,
    score_l_uni:   float,
    tolerance:     float = 0.05,
) -> Dict[str, float]:
    """
    Anchor Question Equity evaluation.

    Tests the hypothesis:
        s_l^{uni} ≈ s_k^{indiv} > s_l^{indiv}

    Parameters
    ----------
    score_k_indiv : correctness score of resource-rich institution k
                    under individual training
    score_l_indiv : correctness score of resource-constrained institution l
                    under individual training
    score_l_uni   : correctness score of institution l after UnifiedEdu
    tolerance     : how close s_l^{uni} must be to s_k^{indiv} to count
                    as "approximately equal"

    Returns
    -------
    dict with keys:
        gap_closed  : float  -- (s_l^{uni} - s_l^{indiv}) / max(s_k^{indiv} - s_l^{indiv}, epsilon)
        hypothesis_met : bool
        scores      : dict of the three input scores
    """
    original_gap = score_k_indiv - score_l_indiv
    gap_closed   = (score_l_uni - score_l_indiv) / max(original_gap, EPSILON)

    hypothesis_met = (
        abs(score_l_uni - score_k_indiv) <= tolerance
        and score_k_indiv > score_l_indiv
    )

    return {
        "gap_closed":       gap_closed,
        "hypothesis_met":   hypothesis_met,
        "score_k_indiv":    score_k_indiv,
        "score_l_indiv":    score_l_indiv,
        "score_l_uni":      score_l_uni,
        "original_gap":     original_gap,
    }