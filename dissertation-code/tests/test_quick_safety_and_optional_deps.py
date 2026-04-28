from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config_loader import Config
from src.main import main
from src.train import train_all_models


def _write_minimal_spy_cache(path: Path, n: int = 5200) -> None:
    idx = pd.bdate_range("2010-01-01", periods=n, freq="C")
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
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


@pytest.fixture()
def base_config(tmp_path: Path) -> Config:
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
        quick_fold_indices=[0, 2],
        inner_cv_splits=3,
        n_iter_hp_search=2,
        n_iter_hp_search_quick=1,
        min_train_rows=200,
        bull_threshold=0.05,
        bear_threshold=-0.05,
        vol_tercile_method="quantile",
        min_folds_per_regime=3,
        transaction_costs_bps=[0],
        risk_free_rate=0.0,
        trading_days_per_year=252,
        results_dir=str(tmp_path / "results"),
        models_dir=str(tmp_path / "models"),
        figures_dir=str(tmp_path / "figures"),
        rf_grid={"n_estimators": [10], "max_depth": [3], "min_samples_split": [5], "min_samples_leaf": [2], "max_features": ["sqrt"]},
        xgb_grid={},
        logistic_grid={"C": [0.1]},
    )


def test_quick_writes_to_subdirs_and_does_not_touch_full_run_dirs(tmp_path: Path, base_config: Config, monkeypatch) -> None:
    # Seed full-run dirs with sentinels.
    results = Path(base_config.results_dir)
    models = Path(base_config.models_dir)
    figures = Path(base_config.figures_dir)
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    (results / "predictions.csv").write_text("SENTINEL_FULL\n", encoding="utf-8")
    (figures / "walkforward_schematic.png").write_bytes(b"SENTINEL_FULL")

    # Create minimal cached data and config.yaml so the CLI path can run.
    # Need enough rows to produce at least one walk-forward fold after
    # feature lookback/dropna and a 5y train window.
    _write_minimal_spy_cache(Path(base_config.data_cache), n=5200)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "ticker: SPY",
                'start_date: "2010-01-01"',
                'end_date: "2026-04-01"',
                f"data_cache: {base_config.data_cache}",
                f"feature_cache: {base_config.feature_cache}",
                "longest_lookback_days: 200",
                "train_years: 5",
                "test_months: 6",
                "test_window_days: 126",
                "purge_days: 200",
                "embargo_days: 10",
                "quick_fold_indices: [0, 2]",
                "inner_cv_splits: 2",
                "n_iter_hp_search: 2",
                "n_iter_hp_search_quick: 1",
                "min_train_rows: 200",
                "bull_threshold: 0.05",
                "bear_threshold: -0.05",
                'vol_tercile_method: "quantile"',
                "min_folds_per_regime: 3",
                "transaction_costs_bps: [0]",
                "risk_free_rate: 0.0",
                "trading_days_per_year: 252",
                f"results_dir: {base_config.results_dir}",
                f"models_dir: {base_config.models_dir}",
                f"figures_dir: {base_config.figures_dir}",
                "rf_grid:",
                "  n_estimators: [10]",
                "  max_depth: [3]",
                "  min_samples_split: [5]",
                "  min_samples_leaf: [2]",
                '  max_features: ["sqrt"]',
                "xgb_grid: {}",
                "logistic_grid:",
                "  C: [0.1]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Avoid any optional heavy bits in quick tests: skip SHAP + figures requiring ffmpeg.
    monkeypatch.setenv("MPLBACKEND", "Agg")

    code = main(["--quick", "--config", str(cfg_path)])
    assert code == 0

    # Full-run sentinels remain untouched.
    assert (results / "predictions.csv").read_text(encoding="utf-8") == "SENTINEL_FULL\n"
    assert (figures / "walkforward_schematic.png").read_bytes() == b"SENTINEL_FULL"

    # Quick outputs exist under subdirectories.
    assert (results / "quick" / "predictions.csv").exists()
    assert (results / "quick" / "dataset_fingerprint.json").exists()


def test_xgboost_optional_import_failure_still_trains_rf_and_baselines(tmp_path: Path, base_config: Config, monkeypatch) -> None:
    # Build a small engineered df and folds by calling the pipeline pieces through main would be slow.
    _write_minimal_spy_cache(Path(base_config.data_cache), n=1600)

    # Prepare an input df that looks like post-features output (train_all_models contract).
    idx = pd.bdate_range("2010-01-01", periods=1600, freq="C")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(index=idx)
    df["log_return"] = rng.normal(0, 0.01, len(idx))
    df["target"] = (pd.Series(df["log_return"], index=idx).shift(-1) > 0).astype(float).fillna(0).astype(int)
    from src.features import FEATURE_COLS

    for c in FEATURE_COLS:
        df[c] = rng.normal(size=len(idx))

    folds = [(np.arange(0, 1200, dtype=np.int64), np.arange(1200, 1326, dtype=np.int64))]

    # Force xgboost import to fail.
    real_import = __import__

    def _blocked_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked_import)

    cfg = replace(base_config, quick_fold_indices=[0])
    train_all_models(df, folds, cfg, quick=True, skip_train=False)

    # RF model and predictions should still exist.
    assert (Path(cfg.models_dir) / "rf_fold_0.pkl").exists()
    preds = pd.read_csv(Path(cfg.results_dir) / "predictions.csv")
    assert set(preds["model"].unique().astype(str)) >= {"majority", "naive", "logistic", "rf"}

