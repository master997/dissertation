"""CLI orchestration for the SPY forecasting pipeline."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
import json
import importlib.util

import pandas as pd

from src.config_loader import load_config
from src.data import download_spy
from src.evaluate import (
    compute_metrics,
    extract_shap,
    regime_stratified_metrics,
    run_significance_tests,
)
from src.features import engineer_features
from src.logging_utils import setup_logging
from src.regimes import classify_regimes
from src.seed import set_seeds
from src.train import train_all_models
from src.trading import run_trading_simulation
from src.validation import walk_forward_splits
from src.visualise import generate_all_figures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPY directional forecasting pipeline")
    p.add_argument("--quick", action="store_true", help="Run subset folds and fewer HP iterations.")
    p.add_argument("--skip-train", action="store_true", help="Skip training; use cached results.")
    p.add_argument(
        "--require-research-deps",
        action="store_true",
        help="Fail fast unless xgboost and shap are installed. Use for full dissertation reproduction.",
    )
    p.add_argument("--seed", type=int, default=None, help="Override config seed.")
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml).",
    )
    return p.parse_args(argv)


def _quick_subdir(path: str) -> str:
    """
    `--quick` must never clobber full-run artefacts.

    Policy: quick runs write into a `quick/` subdirectory under the configured
    output directory (results/models/figures).
    """
    p = Path(path)
    return str(p / "quick")


def _ensure_research_dependencies() -> tuple[bool, list[str]]:
    missing = [name for name in ("xgboost", "shap") if importlib.util.find_spec(name) is None]
    return (len(missing) == 0), missing


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cwd = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = cwd / config_path

    config = load_config(config_path)
    if args.seed is not None:
        config = replace(config, seed=int(args.seed))

    if args.quick:
        config = replace(
            config,
            results_dir=_quick_subdir(config.results_dir),
            models_dir=_quick_subdir(config.models_dir),
            figures_dir=_quick_subdir(config.figures_dir),
        )

    set_seeds(config.seed)
    setup_logging(config.results_dir)
    logger = logging.getLogger("main")
    logger.info("Starting pipeline (quick=%s, skip_train=%s)", args.quick, args.skip_train)

    if args.require_research_deps:
        ok, missing = _ensure_research_dependencies()
        if not ok:
            logger.error(
                "Missing required research dependencies: %s. Install with `pip install -e \".[dev,research]\"`.",
                ", ".join(sorted(missing)),
            )
            return 1

    df_raw = download_spy(config)

    # Dataset fingerprint: proves the run used the shipped CSV.
    cache_path = Path(config.data_cache)
    meta: dict[str, Any] = {
        "ticker": config.ticker,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "data_cache": str(cache_path),
        "rows": int(len(df_raw)),
        "date_min": str(df_raw.index.min()) if not df_raw.empty else None,
        "date_max": str(df_raw.index.max()) if not df_raw.empty else None,
        "sha256": None,
    }
    if cache_path.exists():
        h = hashlib.sha256()
        with cache_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        meta["sha256"] = h.hexdigest()
    out_meta = Path(config.results_dir) / "dataset_fingerprint.json"
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    df = engineer_features(df_raw, config)

    folds = walk_forward_splits(df, config)
    regimes_df = classify_regimes(folds, df, config)

    train_all_models(df, folds, config, quick=args.quick, skip_train=args.skip_train)

    preds_path = Path(config.results_dir) / "predictions.csv"
    if not preds_path.exists():
        logger.error("predictions.csv missing — cannot evaluate.")
        return 1

    compute_metrics(config)
    run_significance_tests(config)
    regime_stratified_metrics(config)

    preds = pd.read_csv(preds_path, parse_dates=["date"])
    extract_shap(df, folds, config, regimes_df, quick=args.quick)

    run_trading_simulation(preds, df, folds, config)

    generate_all_figures(config)

    logger.info("Pipeline finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
