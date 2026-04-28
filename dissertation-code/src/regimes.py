"""
Return and volatility regime labels per walk-forward fold.

Reference: Lo (2004) — adaptive markets hypothesis (AMH), evaluated via regime splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.config_loader import Config

logger = logging.getLogger(__name__)


def classify_regimes(
    folds: list[tuple[np.ndarray, np.ndarray]],
    df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Label each fold's test window by cumulative log return (bull/bear/sideways)
    and realised volatility tercile (low/mid/high).
    """
    if config.vol_tercile_method.lower() != "quantile":
        raise ValueError(
            "Unsupported vol_tercile_method="
            f"{config.vol_tercile_method!r}; only 'quantile' is implemented."
        )

    lr = df["log_return"].astype(float)
    rows = []
    for fold_id, (_, test_idx) in enumerate(folds):
        test_lr = lr.iloc[test_idx]
        realised_return = float(test_lr.sum())
        realised_vol = float(test_lr.std() * np.sqrt(config.trading_days_per_year))
        t0 = df.index[int(test_idx[0])]
        t1 = df.index[int(test_idx[-1])]
        if realised_return > config.bull_threshold:
            rreg = "bull"
        elif realised_return < config.bear_threshold:
            rreg = "bear"
        else:
            rreg = "sideways"
        rows.append(
            {
                "fold_id": fold_id,
                "test_start": t0,
                "test_end": t1,
                "realised_return": realised_return,
                "realised_vol": realised_vol,
                "return_regime": rreg,
            }
        )

    out = pd.DataFrame(rows)
    vols = out["realised_vol"].to_numpy()
    q1, q2 = np.quantile(vols, [1 / 3, 2 / 3])

    def vol_bucket(v: float) -> str:
        if v <= q1:
            return "low"
        if v <= q2:
            return "mid"
        return "high"

    out["vol_regime"] = out["realised_vol"].map(vol_bucket)

    regime_counts = out["return_regime"].value_counts().reindex(
        ["bull", "bear", "sideways"], fill_value=0
    )
    logger.info("Return regime distribution:\n%s", regime_counts)
    if regime_counts.min() < config.min_folds_per_regime:
        logger.warning(
            "Regime '%s' has only %d folds. Treat regime-stratified metrics as descriptive "
            "when a regime has fewer than %d folds, and report this as a limitation.",
            regime_counts.idxmin(),
            int(regime_counts.min()),
            config.min_folds_per_regime,
        )

    out_path = Path(config.results_dir) / "regime_assignments.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return out
