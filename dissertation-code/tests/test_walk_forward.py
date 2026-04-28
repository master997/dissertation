"""Walk-forward splitter invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config_loader import Config
from src.validation import walk_forward_splits


@pytest.fixture()
def wf_config(tmp_path) -> Config:
    return Config(
        seed=42,
        ticker="SPY",
        start_date="2010-01-01",
        end_date="2026-04-01",
        data_cache=str(tmp_path / "spy_raw.csv"),
        feature_cache=str(tmp_path / "spy_features.csv"),
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
        results_dir=str(tmp_path / "results"),
        models_dir=str(tmp_path / "models"),
        figures_dir=str(tmp_path / "figures"),
        rf_grid={},
        xgb_grid={},
        logistic_grid={},
    )


def _dummy_df(start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end, freq="C")
    return pd.DataFrame({"Close": np.linspace(100, 200, len(idx))}, index=idx)


def test_no_overlap_and_purge_gap(wf_config: Config) -> None:
    df = _dummy_df("2010-01-01", "2026-04-01")
    folds = walk_forward_splits(df, wf_config)
    assert len(folds) >= 15
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert int(np.max(train_idx)) + wf_config.purge_days <= int(np.min(test_idx))


def test_chronological_order(wf_config: Config) -> None:
    df = _dummy_df("2010-01-01", "2026-04-01")
    folds = walk_forward_splits(df, wf_config)
    test_starts = [df.index[int(t[1][0])] for t in folds]
    assert test_starts == sorted(test_starts)


def test_fold_count_minimum(wf_config: Config) -> None:
    df = _dummy_df("2010-01-01", "2026-04-01")
    folds = walk_forward_splits(df, wf_config)
    assert len(folds) >= 15


def test_embargo_is_calendar_advance_then_snap_to_trading_day(wf_config: Config) -> None:
    df = _dummy_df("2010-01-01", "2026-04-01")
    folds = walk_forward_splits(df, wf_config)

    # Recreate the splitter's cursor semantics: advance by DateOffset(months, days),
    # then choose first available trading row at/after that timestamp.
    dates = df.index.sort_values()
    cursor = dates[0] + pd.DateOffset(years=wf_config.train_years)

    for _i, (_tr, te) in enumerate(folds):
        test_first = dates[int(te[0])]
        assert test_first >= cursor
        cursor = cursor + pd.DateOffset(months=wf_config.test_months, days=wf_config.embargo_days)
