"""
evaluation/statistical.py

Statistical validation utilities:
  - Paired Wilcoxon signed-rank test with Bonferroni correction
  - Bootstrap confidence intervals
  - Cohen's d effect size
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def wilcoxon_test(
    scores_indiv: List[float],
    scores_uni:   List[float],
    alpha:        float = 0.05,
) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank test:
        H0: mu_{uni} = mu_{indiv}
        H1: mu_{uni} > mu_{indiv}  (one-sided)

    Parameters
    ----------
    scores_indiv : per-sample quality scores under individual training
    scores_uni   : per-sample quality scores under UnifiedEdu

    Returns
    -------
    dict with keys: statistic, p_value, significant, cohens_d
    """
    a = np.array(scores_indiv, dtype=float)
    b = np.array(scores_uni,   dtype=float)

    if len(a) < 2 or len(b) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "cohens_d": 0.0}

    stat, p_two = stats.wilcoxon(a, b, alternative="less")
    # alternative="less" tests H1: a < b, i.e. indiv < uni
    significant = p_two < alpha

    # Cohen's d
    diff = b - a
    d = float(diff.mean() / (diff.std() + 1e-9))

    return {
        "statistic":   float(stat),
        "p_value":     float(p_two),
        "significant": significant,
        "cohens_d":    d,
    }


def bonferroni_correction(
    p_values: List[float],
    alpha:    float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply Bonferroni correction to a list of p-values.

    Returns
    -------
    (corrected_p_values, reject_null_list)
    """
    n = len(p_values)
    corrected = [min(1.0, p * n) for p in p_values]
    reject    = [p < alpha for p in corrected]
    return corrected, reject


def bootstrap_ci(
    scores:      List[float],
    n_bootstrap: int   = 10_000,
    ci:          float = 0.95,
    seed:        int   = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    Returns
    -------
    (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=float)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return float(arr.mean()), lo, hi