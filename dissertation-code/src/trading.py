"""
Trading simulation: long-or-flat with transaction costs and fold boundaries.

References: Bacon (2008); Bailey & López de Prado (2014).
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


def _simulate_fold(
    positions: np.ndarray,
    next_day_log_returns: np.ndarray,
    *,
    fold_entry_long: bool,
    fold_exit_long: bool,
    cost_bps: float,
) -> np.ndarray:
    """next_day_log_returns[i] is P&L from holding long over i→i+1 when position[i]==1."""
    pos = positions.astype(float)
    r = np.zeros_like(next_day_log_returns, dtype=float)
    c = cost_bps / 10000.0

    if fold_entry_long and len(pos) and pos[0] > 0:
        r[0] -= c

    prev = 0.0
    for i in range(len(pos)):
        if i > 0 and pos[i] != prev:
            r[i] -= c
        prev = pos[i]
        r[i] += pos[i] * next_day_log_returns[i]

    if fold_exit_long and len(pos) and pos[-1] > 0:
        r[-1] -= c

    return r


def run_trading_simulation(
    preds: pd.DataFrame,
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: Config,
) -> pd.DataFrame:
    """
    Long-or-flat from predictions; sweep transaction costs.

    preds must include columns: fold_id, date, model, y_pred.
    df must include log_return indexed by date.
    """
    df = df.sort_index().copy()
    df["lr_next"] = df["log_return"].shift(-1)

    rows: list[dict] = []
    models = sorted(preds["model"].unique())

    for model in models:
        for cost_bps in config.transaction_costs_bps:
            strat_lr: list[float] = []

            for fold_id, (_, test_idx) in enumerate(folds):
                pm = preds[(preds["fold_id"] == fold_id) & (preds["model"] == model)]
                if pm.empty:
                    continue
                pm = pm.sort_values("date").copy()
                merged = pm.merge(
                    df[["lr_next"]],
                    left_on="date",
                    right_index=True,
                    how="left",
                )
                next_lr = merged["lr_next"].to_numpy(dtype=float)
                pred_dir = merged["y_pred"].to_numpy(dtype=int)
                positions = np.where(pred_dir == 1, 1.0, 0.0)

                fold_r = _simulate_fold(
                    positions,
                    next_lr,
                    fold_entry_long=bool(positions[0] > 0) if len(positions) else False,
                    fold_exit_long=bool(positions[-1] > 0) if len(positions) else False,
                    cost_bps=float(cost_bps),
                )
                strat_lr.extend(fold_r[~np.isnan(fold_r)].tolist())

            if not strat_lr:
                continue

            lr_series = np.asarray(strat_lr, dtype=float)
            cum_log = float(np.sum(lr_series))
            cum_simple = float(np.prod(np.exp(lr_series)) - 1.0)
            mean_daily = float(np.mean(lr_series))
            vol_daily = float(np.std(lr_series, ddof=1)) if len(lr_series) > 1 else 0.0
            ann_ret = mean_daily * config.trading_days_per_year
            ann_vol = vol_daily * np.sqrt(config.trading_days_per_year)
            sharpe = (
                (ann_ret - config.risk_free_rate) / ann_vol if ann_vol > 1e-12 else float("nan")
            )

            equity = np.cumprod(np.exp(lr_series))
            peak = np.maximum.accumulate(equity)
            max_dd = float(np.min(equity / peak - 1.0))
            calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else float("nan")

            wins = lr_series[lr_series > 0]
            losses = lr_series[lr_series < 0]

            rows.append(
                {
                    "model": model,
                    "cost_bps": cost_bps,
                    "cumulative_log_return": cum_log,
                    "cumulative_simple_return": cum_simple,
                    "annualised_return": ann_ret,
                    "annualised_vol": ann_vol,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "calmar": calmar,
                    "win_rate": float(np.mean(lr_series > 0)),
                    "avg_win": float(np.mean(wins)) if len(wins) else 0.0,
                    "avg_loss": float(np.mean(losses)) if len(losses) else 0.0,
                    "win_loss_ratio": (
                        float(np.mean(wins) / abs(np.mean(losses)))
                        if len(wins) and len(losses)
                        else float("nan")
                    ),
                    "turnover": float("nan"),
                }
            )

    # Buy-and-hold on same concatenated calendar as first model path (union of test dates)
    test_dates = sorted(pd.to_datetime(preds["date"]).unique())
    bh_base = df.reindex(test_dates)

    for cost_bps in config.transaction_costs_bps:
        next_lr = bh_base["lr_next"].to_numpy(dtype=float)
        pos = np.ones(len(next_lr))
        r_bh = _simulate_fold(
            pos,
            next_lr,
            fold_entry_long=True,
            fold_exit_long=True,
            cost_bps=float(cost_bps),
        )
        r_bh = r_bh[~np.isnan(r_bh)]
        if len(r_bh) == 0:
            continue
        mean_daily = float(np.mean(r_bh))
        vol_daily = float(np.std(r_bh, ddof=1)) if len(r_bh) > 1 else 0.0
        ann_ret = mean_daily * config.trading_days_per_year
        ann_vol = vol_daily * np.sqrt(config.trading_days_per_year)
        sharpe = (
            (ann_ret - config.risk_free_rate) / ann_vol if ann_vol > 1e-12 else float("nan")
        )
        equity = np.cumprod(np.exp(r_bh))
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min(equity / peak - 1.0))
        rows.append(
            {
                "model": "buy_hold",
                "cost_bps": cost_bps,
                "cumulative_log_return": float(np.sum(r_bh)),
                "cumulative_simple_return": float(np.prod(np.exp(r_bh)) - 1.0),
                "annualised_return": ann_ret,
                "annualised_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "calmar": ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else float("nan"),
                "win_rate": float(np.mean(r_bh > 0)),
                "avg_win": float(np.mean(r_bh[r_bh > 0])) if np.any(r_bh > 0) else 0.0,
                "avg_loss": float(np.mean(r_bh[r_bh < 0])) if np.any(r_bh < 0) else 0.0,
                "win_loss_ratio": float("nan"),
                "turnover": float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(config.results_dir) / "trading_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return out
