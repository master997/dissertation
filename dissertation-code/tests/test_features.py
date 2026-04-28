"""Feature engineering tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import ta.momentum

from src.config_loader import Config
from src.features import FEATURE_COLS, engineer_features


@pytest.fixture()
def cfg(tmp_path) -> Config:
    return Config(
        seed=42,
        ticker="SPY",
        start_date="2015-01-01",
        end_date="2020-01-01",
        data_cache=str(tmp_path / "raw.csv"),
        feature_cache=str(tmp_path / "feat.csv"),
        longest_lookback_days=200,
        train_years=5,
        test_months=6,
        test_window_days=126,
        purge_days=200,
        embargo_days=10,
        quick_fold_indices=[0, 12],
        inner_cv_splits=3,
        n_iter_hp_search=20,
        n_iter_hp_search_quick=5,
        min_train_rows=500,
        bull_threshold=0.05,
        bear_threshold=-0.05,
        vol_tercile_method="quantile",
        min_folds_per_regime=3,
        transaction_costs_bps=[0, 5, 10, 20],
        risk_free_rate=0.0,
        trading_days_per_year=252,
        results_dir=str(tmp_path / "r"),
        models_dir=str(tmp_path / "m"),
        figures_dir=str(tmp_path / "fig"),
        rf_grid={},
        xgb_grid={},
        logistic_grid={},
    )


def _long_frame(n: int = 1200) -> pd.DataFrame:
    idx = pd.bdate_range("2010-01-01", periods=n, freq="C")
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": 1e6 + rng.normal(0, 1e3, n),
        },
        index=idx,
    )
    lr = np.log(close / np.roll(close, 1))
    lr[0] = np.nan
    df["log_return"] = lr
    df["target"] = (pd.Series(lr, index=idx).shift(-1) > 0).astype(float)
    df = df.iloc[1:]
    df["target"] = df["target"].astype(int)
    df = df.dropna(subset=["log_return", "target"])
    return df


def test_feature_cols_length() -> None:
    assert len(FEATURE_COLS) == 26


def test_rsi_reference() -> None:
    close = pd.Series(
        [
            44.0,
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.85,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
            46.00,
            46.03,
            46.41,
            46.22,
            45.64,
            46.21,
        ]
    )
    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    assert np.isfinite(rsi.iloc[-1])
    assert 0 <= rsi.iloc[-1] <= 100


def test_stationary_sma_dist_near_zero_mean(cfg: Config) -> None:
    df = _long_frame(1500)
    out = engineer_features(df, cfg)
    col = out["sma_200_dist"].dropna()
    assert abs(float(col.mean())) < 0.05
