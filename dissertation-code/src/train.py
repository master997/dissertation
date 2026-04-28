"""
Train RF / XGBoost with nested time-series search; persist fold artefacts.

References: Breiman (2001); Chen & Guestrin (2016); López de Prado (2018).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

if TYPE_CHECKING:
    from src.config_loader import Config

from src.baselines import logistic_baseline, majority_baseline, naive_prevday_baseline
from src.features import FEATURE_COLS

logger = logging.getLogger(__name__)


def _rf_grid(config: Config) -> dict[str, list[Any]]:
    return {k: list(v) for k, v in config.rf_grid.items()}


def _xgb_grid(config: Config) -> dict[str, list[Any]]:
    return {k: list(v) for k, v in config.xgb_grid.items()}


def train_all_models(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: Config,
    *,
    quick: bool = False,
    skip_train: bool = False,
) -> pd.DataFrame:
    """
    Train RF, XGBoost, logistic; compute majority and naive baselines per fold.

    Writes models/*.pkl, results/fold_results.csv, results/predictions.csv.
    """
    models_dir = Path(config.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for c in FEATURE_COLS + ["target", "log_return"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if skip_train:
        fr_path = results_dir / "fold_results.csv"
        if fr_path.exists():
            logger.info("skip_train: loading existing %s", fr_path)
            return pd.read_csv(fr_path)
        raise FileNotFoundError("skip_train set but results/fold_results.csv not found.")

    fold_ids: set[int] | None
    if quick and config.quick_fold_indices:
        fold_ids = set(int(x) for x in config.quick_fold_indices)
    else:
        fold_ids = None

    if fold_ids is not None:
        bad = sorted([i for i in fold_ids if i < 0 or i >= len(folds)])
        if bad:
            raise ValueError(
                f"quick_fold_indices contains out-of-range fold ids: {bad}. "
                f"Available folds: 0..{max(len(folds)-1, 0)}"
            )

    # Safety: prevent --quick from overwriting a full run's artefacts.
    # A quick run trains only a subset of folds; overwriting predictions.csv would silently destroy results.
    if quick:
        existing_preds = results_dir / "predictions.csv"
        if existing_preds.exists() and fold_ids is not None:
            try:
                ex = pd.read_csv(existing_preds, usecols=["fold_id"])
                existing_fold_count = int(ex["fold_id"].nunique())
            except Exception:
                existing_fold_count = -1
            planned_fold_count = len(fold_ids)
            if existing_fold_count > planned_fold_count:
                raise RuntimeError(
                    f"--quick would overwrite an existing full run (existing folds={existing_fold_count}, "
                    f"planned folds={planned_fold_count}). Use --skip-train to reuse, or run quick in a "
                    f"separate clean folder/results_dir."
                )
    n_iter = config.n_iter_hp_search_quick if quick else config.n_iter_hp_search
    inner = TimeSeriesSplit(n_splits=config.inner_cv_splits)
    n_jobs = 1 if quick else -1

    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    # Optional dependency: XGBoost can be painful on some machines (OpenMP issues on macOS).
    # The pipeline should still be runnable and evaluable without it.
    xgb_available = True
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:
        xgb_available = False
        xgb = None  # type: ignore[assignment]
        logger.warning("XGBoost unavailable (%s). Proceeding without XGBoost.", exc)

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        if fold_ids is not None and fold_id not in fold_ids:
            continue

        X_train = df.iloc[train_idx][FEATURE_COLS].to_numpy(dtype=float)
        y_train = df.iloc[train_idx]["target"].to_numpy(dtype=int)
        X_test = df.iloc[test_idx][FEATURE_COLS].to_numpy(dtype=float)
        y_test = df.iloc[test_idx]["target"].to_numpy(dtype=int)
        lr_test = df.iloc[test_idx]["log_return"].to_numpy(dtype=float)

        if len(X_train) < config.min_train_rows:
            logger.warning(
                "Fold %d: only %d training rows after purging (minimum %d). Skipping fold.",
                fold_id,
                len(X_train),
                config.min_train_rows,
            )
            continue

        dates_test = df.index[test_idx]

        # --- Majority ---
        maj_p, maj_pr = majority_baseline(y_train, len(y_test))
        fold_rows.append(
            {
                "fold_id": fold_id,
                "model": "majority",
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "best_params": json.dumps({}),
            }
        )
        for i in range(len(y_test)):
            pred_rows.append(
                {
                    "fold_id": fold_id,
                    "model": "majority",
                    "date": dates_test[i],
                    "y_true": int(y_test[i]),
                    "y_pred": int(maj_p[i]),
                    "y_proba": float(maj_pr[i]),
                }
            )

        # --- Naive ---
        nv_p, nv_pr = naive_prevday_baseline(lr_test)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "model": "naive",
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "best_params": json.dumps({}),
            }
        )
        for i in range(len(y_test)):
            pred_rows.append(
                {
                    "fold_id": fold_id,
                    "model": "naive",
                    "date": dates_test[i],
                    "y_true": int(y_test[i]),
                    "y_pred": int(nv_p[i]),
                    "y_proba": float(nv_pr[i]),
                }
            )

        # --- Logistic ---
        lg_p, lg_pr, lg_bp = logistic_baseline(X_train, y_train, X_test, config, seed=config.seed)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "model": "logistic",
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "best_params": json.dumps(lg_bp),
            }
        )
        for i in range(len(y_test)):
            pred_rows.append(
                {
                    "fold_id": fold_id,
                    "model": "logistic",
                    "date": dates_test[i],
                    "y_true": int(y_test[i]),
                    "y_pred": int(lg_p[i]),
                    "y_proba": float(lg_pr[i]),
                }
            )

        # --- Random Forest ---
        rf_clf = RandomForestClassifier(class_weight="balanced", random_state=config.seed)
        rf_search = RandomizedSearchCV(
            rf_clf,
            _rf_grid(config),
            n_iter=n_iter,
            cv=inner,
            scoring="roc_auc",
            random_state=config.seed,
            n_jobs=n_jobs,
            refit=True,
        )
        with parallel_backend("threading"):
            rf_search.fit(X_train, y_train)
        rf_pred = rf_search.predict(X_test)
        rf_pr = rf_search.predict_proba(X_test)[:, 1]
        with open(models_dir / f"rf_fold_{fold_id}.pkl", "wb") as fh:
            pickle.dump(rf_search.best_estimator_, fh)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "model": "rf",
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "best_params": json.dumps(rf_search.best_params_),
            }
        )
        for i in range(len(y_test)):
            pred_rows.append(
                {
                    "fold_id": fold_id,
                    "model": "rf",
                    "date": dates_test[i],
                    "y_true": int(y_test[i]),
                    "y_pred": int(rf_pred[i]),
                    "y_proba": float(rf_pr[i]),
                }
            )

        # --- XGBoost ---
        if not xgb_available:
            continue
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        spw = neg / pos if pos > 0 else 1.0

        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config.seed,
            n_jobs=n_jobs,
            scale_pos_weight=spw,
        )
        xgb_search = RandomizedSearchCV(
            xgb_clf,
            _xgb_grid(config),
            n_iter=n_iter,
            cv=inner,
            scoring="roc_auc",
            random_state=config.seed,
            n_jobs=n_jobs,
            refit=True,
        )
        with parallel_backend("threading"):
            xgb_search.fit(X_train, y_train)
        xgb_pred = xgb_search.predict(X_test)
        xgb_pr = xgb_search.predict_proba(X_test)[:, 1]
        with open(models_dir / f"xgb_fold_{fold_id}.pkl", "wb") as fh:
            pickle.dump(xgb_search.best_estimator_, fh)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "model": "xgb",
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "best_params": json.dumps(xgb_search.best_params_),
            }
        )
        for i in range(len(y_test)):
            pred_rows.append(
                {
                    "fold_id": fold_id,
                    "model": "xgb",
                    "date": dates_test[i],
                    "y_true": int(y_test[i]),
                    "y_pred": int(xgb_pred[i]),
                    "y_proba": float(xgb_pr[i]),
                }
            )

    fr = pd.DataFrame(fold_rows)
    fr.to_csv(results_dir / "fold_results.csv", index=False)
    pr = pd.DataFrame(pred_rows)
    pr.to_csv(results_dir / "predictions.csv", index=False)
    logger.info("Wrote fold_results and predictions (%d prediction rows).", len(pr))
    return fr
