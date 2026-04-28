"""
Technical feature engineering (26 features).

References: Htun et al. (2023); Tsay (2010); López de Prado (2018) ch. 5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume

if TYPE_CHECKING:
    from src.config_loader import Config

logger = logging.getLogger(__name__)

_WARMUP_TOLERANCE_ROWS = 5

FEATURE_COLS = [
    # Momentum (6)
    "rsi_14",
    "stoch_k",
    "stoch_d",
    "macd_signal",
    "macd_hist",
    "roc_14",
    # Trend (7)
    "sma_50_dist",
    "sma_100_dist",
    "sma_200_dist",
    "ema_50_dist",
    "ema_100_dist",
    "ema_200_dist",
    "adx_14",
    # Volatility (5)
    "bb_upper_dist",
    "bb_lower_dist",
    "bb_width_20",
    "atr_14",
    "rolling_std_20",
    # Volume (3)
    "obv_diff",
    "vwap_dist",
    "cmf_20",
    # Autoregressive (5)
    "lag_ret_1",
    "lag_ret_2",
    "lag_ret_3",
    "lag_ret_4",
    "lag_ret_5",
]
assert len(FEATURE_COLS) == 26, f"Expected 26 features, got {len(FEATURE_COLS)}"


def engineer_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Compute 26 technical / autoregressive features; drop rows with NaN; assert column set.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)

    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    vol = out["Volume"].astype(float)
    log_ret = out["log_return"].astype(float)

    out["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3
    )
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    macd = ta.trend.MACD(close=close)
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()
    out["roc_14"] = ta.momentum.ROCIndicator(close=close, window=14).roc()

    for w, name in [(50, "sma_50_dist"), (100, "sma_100_dist"), (200, "sma_200_dist")]:
        sma = ta.trend.SMAIndicator(close=close, window=w).sma_indicator()
        out[name] = (close - sma) / sma
    for w, name in [(50, "ema_50_dist"), (100, "ema_100_dist"), (200, "ema_200_dist")]:
        ema = ta.trend.EMAIndicator(close=close, window=w).ema_indicator()
        out[name] = (close - ema) / ema
    out["adx_14"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()

    bb = ta.volatility.BollingerBands(close=close, window=20)
    hband = bb.bollinger_hband()
    lband = bb.bollinger_lband()
    out["bb_upper_dist"] = (close - hband) / hband
    out["bb_lower_dist"] = (close - lband) / lband
    out["bb_width_20"] = bb.bollinger_wband()
    out["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    out["rolling_std_20"] = log_ret.rolling(20).std()

    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
    out["obv_diff"] = obv.diff()
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=vol, window=14
    ).volume_weighted_average_price()
    out["vwap_dist"] = (close - vwap) / vwap
    out["cmf_20"] = ta.volume.ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=vol, window=20
    ).chaikin_money_flow()

    for i in range(1, 6):
        out[f"lag_ret_{i}"] = log_ret.shift(i)

    cols_needed = FEATURE_COLS + ["log_return", "target"]
    missing = set(FEATURE_COLS) - set(out.columns)
    assert not missing, f"Missing features after engineering: {missing}"

    rows_before_drop = len(out)
    out = out.dropna(subset=cols_needed)
    rows_dropped = rows_before_drop - len(out)
    warmup_msg = "Feature warm-up dropped %d rows (configured longest_lookback_days=%d)."
    if rows_dropped > config.longest_lookback_days + _WARMUP_TOLERANCE_ROWS:
        logger.warning(warmup_msg, rows_dropped, config.longest_lookback_days)
    else:
        logger.info(warmup_msg, rows_dropped, config.longest_lookback_days)
    # log_return kept for regimes/trading; models use FEATURE_COLS only.
    out = out[FEATURE_COLS + ["log_return", "target"]]

    cache = Path(config.feature_cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    logger.info("Wrote feature cache %s (%d rows).", cache, len(out))

    return out
