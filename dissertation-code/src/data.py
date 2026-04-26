"""
SPY data ingestion and target construction.

References: Tsay (2010) ch. 2 — Analysis of Financial Time Series.
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


def download_spy(config: Config) -> pd.DataFrame:
    """
    Download adjusted SPY OHLCV, cache to CSV, build log_return and next-day direction target.

    Target: target[t] = 1 iff log(close[t+1]/close[t]) > 0 (no lookahead in features at t).
    """
    cache_path = Path(config.data_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        logger.info("Loaded cached data from %s (%d rows).", cache_path, len(df))
    else:
        import yfinance as yf

        df = yf.download(
            config.ticker,
            start=config.start_date,
            end=config.end_date,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise RuntimeError("yfinance returned no data for the given range.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=lambda c: str(c).strip())
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.to_csv(cache_path)
        logger.info("Saved download to %s (%d rows).", cache_path, len(df))

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=lambda c: str(c).strip())
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns after download: {missing}")

    close = df["Close"].astype(float)
    df["log_return"] = np.log(close / close.shift(1))
    # Next-trading-day direction label.
    # Important: the terminal row has no next-day return; keep it as NaN so it gets dropped.
    lr_next = df["log_return"].shift(-1)
    df["target"] = (lr_next > 0).astype(float)
    df.loc[lr_next.isna(), "target"] = np.nan

    df = df.dropna(subset=["log_return", "target"])
    df["target"] = df["target"].astype(int)

    keep = ["Open", "High", "Low", "Close", "Volume", "log_return", "target"]
    out = df[keep].copy()
    return out
