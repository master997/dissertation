"""
Statistical tests for predictive accuracy comparisons.

References: Diebold & Mariano (1995); scipy/statsmodels conventions.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac

logger = logging.getLogger(__name__)


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    max_lag: int = 5,
) -> tuple[float, float]:
    """
    Two-sided Diebold-Mariano test for equal predictive accuracy.
    Returns (dm_statistic, two_sided_p_value).
    """
    d = loss_a - loss_b
    t = len(d)
    d_bar = np.mean(d)
    x = np.ones((t, 1))
    ols_fit = OLS(d, x).fit()
    hac_var = cov_hac(ols_fit, nlags=max_lag)[0, 0]
    if hac_var <= 0 or not np.isfinite(hac_var):
        return float("nan"), float("nan")
    # cov_hac returns the HAC variance of the sample mean d̄ (not of d_t),
    # so we must not divide by t again.
    dm_stat = d_bar / np.sqrt(hac_var)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def holm_corrected_ttests(
    fold_acc: dict[str, list[float]],
    comparisons: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """Paired t-tests on per-fold accuracies with Holm correction."""
    raw_p: list[float] = []
    rows: list[dict[str, object]] = []
    for a, b in comparisons:
        x = np.asarray(fold_acc[a], dtype=float)
        y = np.asarray(fold_acc[b], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            p = 1.0
            tstat = float("nan")
        else:
            tstat, p = stats.ttest_rel(x, y)
        raw_p.append(float(p))
        rows.append(
            {
                "comparison": f"{a}_vs_{b}",
                "test": "paired_t_holm",
                "statistic": float(tstat),
                "p_value": float(p),
                "p_value_corrected": float("nan"),
            }
        )
    _, p_corr, _, _ = multipletests(raw_p, method="holm")
    for i, row in enumerate(rows):
        row["p_value_corrected"] = float(p_corr[i])
    return pd.DataFrame(rows)
