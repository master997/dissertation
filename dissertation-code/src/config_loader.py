"""Load project configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _replace_null(obj: Any) -> Any:
    """Convert YAML null in nested dicts to Python None for sklearn params."""
    if obj is None or (isinstance(obj, str) and obj.lower() == "null"):
        return None
    if isinstance(obj, dict):
        return {k: _replace_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_null(x) for x in obj]
    return obj


@dataclass
class Config:
    """Application configuration (single source of truth: config.yaml)."""

    seed: int
    ticker: str
    start_date: str
    end_date: str
    data_cache: str
    feature_cache: str
    longest_lookback_days: int
    train_years: int
    test_months: int
    test_window_days: int
    purge_days: int
    embargo_days: int
    quick_fold_indices: list[int]
    inner_cv_splits: int
    n_iter_hp_search: int
    n_iter_hp_search_quick: int
    min_train_rows: int
    bull_threshold: float
    bear_threshold: float
    vol_tercile_method: str
    min_folds_per_regime: int
    transaction_costs_bps: list[int]
    risk_free_rate: float
    trading_days_per_year: int
    results_dir: str
    models_dir: str
    figures_dir: str
    rf_grid: dict[str, list[Any]] = field(default_factory=dict)
    xgb_grid: dict[str, list[Any]] = field(default_factory=dict)
    logistic_grid: dict[str, list[float]] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML path (default: config.yaml next to cwd or package root)."""
    if path is None:
        path = Path.cwd() / "config.yaml"
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _replace_null(raw)

    return Config(
        seed=int(raw["seed"]),
        ticker=str(raw["ticker"]),
        start_date=str(raw["start_date"]),
        end_date=str(raw["end_date"]),
        data_cache=str(raw["data_cache"]),
        feature_cache=str(raw["feature_cache"]),
        longest_lookback_days=int(raw["longest_lookback_days"]),
        train_years=int(raw["train_years"]),
        test_months=int(raw["test_months"]),
        test_window_days=int(raw["test_window_days"]),
        purge_days=int(raw["purge_days"]),
        embargo_days=int(raw["embargo_days"]),
        quick_fold_indices=list(raw["quick_fold_indices"]),
        inner_cv_splits=int(raw["inner_cv_splits"]),
        n_iter_hp_search=int(raw["n_iter_hp_search"]),
        n_iter_hp_search_quick=int(raw["n_iter_hp_search_quick"]),
        min_train_rows=int(raw["min_train_rows"]),
        bull_threshold=float(raw["bull_threshold"]),
        bear_threshold=float(raw["bear_threshold"]),
        vol_tercile_method=str(raw["vol_tercile_method"]),
        min_folds_per_regime=int(raw["min_folds_per_regime"]),
        transaction_costs_bps=list(raw["transaction_costs_bps"]),
        risk_free_rate=float(raw["risk_free_rate"]),
        trading_days_per_year=int(raw["trading_days_per_year"]),
        results_dir=str(raw["results_dir"]),
        models_dir=str(raw["models_dir"]),
        figures_dir=str(raw["figures_dir"]),
        rf_grid=dict(raw["rf_grid"]),
        xgb_grid=dict(raw["xgb_grid"]),
        logistic_grid=dict(raw["logistic_grid"]),
    )
