"""Target construction correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config_loader import Config
from src.data import download_spy


@pytest.fixture()
def cfg(tmp_path) -> Config:
    return Config(
        seed=42,
        ticker="SPY",
        start_date="2020-01-01",
        end_date="2020-06-01",
        data_cache=str(tmp_path / "spy_raw.csv"),
        feature_cache=str(tmp_path / "f.csv"),
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


def _write_synth_cache(path, n: int = 500) -> None:
    idx = pd.bdate_range("2015-01-01", periods=n, freq="C")
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": 1e6,
        },
        index=idx,
    )
    df.to_csv(path)


def test_target_matches_shifted_return(cfg: Config) -> None:
    _write_synth_cache(cfg.data_cache, n=300)
    df = download_spy(cfg)

    # Compute expected target from the raw cached OHLCV (not from already-processed df).
    raw = pd.read_csv(cfg.data_cache, index_col=0, parse_dates=True).sort_index()
    close = raw["Close"].astype(float)
    lr = np.log(close / close.shift(1))
    lr_next = lr.shift(-1)
    expected_target = (lr_next > 0).astype(float)
    expected_target.loc[lr_next.isna()] = np.nan

    expected = pd.DataFrame({"log_return": lr, "target": expected_target}).dropna(subset=["log_return", "target"])
    expected["target"] = expected["target"].astype(int)

    assert df.index.equals(expected.index)
    assert df["target"].dtype == int
    assert (df["target"] == expected["target"]).all()
    assert df["target"].notna().all()


def test_target_mean_band_long_sample(cfg: Config) -> None:
    """Bernoulli mean near 0.5 for symmetric lognormal steps; SPY live data should fall in [0.50, 0.56]."""
    _write_synth_cache(cfg.data_cache, n=20_000)
    df = download_spy(cfg)
    m = float(df["target"].mean())
    assert 0.48 <= m <= 0.52


def test_target_independent_of_same_row_forward_return_component(cfg: Config) -> None:
    """Target uses r_{t+1}; same-row log_return is r_t — must not determine target alone."""
    _write_synth_cache(cfg.data_cache, n=400)
    df = download_spy(cfg)
    # Correlation between target and contemporaneous log_return should not be perfect
    # (would indicate accidental same-day leakage).
    rho = df["target"].corr((df["log_return"] > 0).astype(int))
    assert rho < 0.999

