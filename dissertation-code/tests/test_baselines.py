"""Baseline predictor tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import logistic_baseline, majority_baseline, naive_prevday_baseline
from src.config_loader import Config


@pytest.fixture()
def cfg(tmp_path) -> Config:
    return Config(
        seed=42,
        ticker="SPY",
        start_date="2010-01-01",
        end_date="2020-01-01",
        data_cache=str(tmp_path / "raw.csv"),
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
        min_train_rows=50,
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
        logistic_grid={"C": [0.01, 0.1, 1.0, 10.0]},
    )


def test_majority_all_ones() -> None:
    y_train = np.array([1, 1, 1, 0])
    pred, _ = majority_baseline(y_train, n_test=5)
    assert np.all(pred == 1)


def test_logistic_probabilities_bounded(cfg: Config) -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 5))
    y_train = (rng.random(200) > 0.48).astype(int)
    X_test = rng.normal(size=(40, 5))
    pred, proba, _best = logistic_baseline(X_train, y_train, X_test, cfg, seed=42)
    assert proba.min() >= 0 and proba.max() <= 1
    assert pred.shape == (40,)


def test_naive_matches_today_sign() -> None:
    lr = np.array([-0.01, 0.02, -0.0, 0.03])
    pred, proba = naive_prevday_baseline(lr)
    assert list(pred) == [0, 1, 0, 1]
    assert np.all(proba == 0.5)
