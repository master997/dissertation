"""
Baseline predictors for comparison.

Majority class, naive previous-day direction, logistic regression with time-series CV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from src.config_loader import Config


@dataclass
class FoldPredictions:
    fold_id: int
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray


def majority_baseline(y_train: np.ndarray, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    vals, counts = np.unique(y_train, return_counts=True)
    maj = int(vals[np.argmax(counts)])
    p_up = float(np.mean(y_train == 1))
    pred = np.full(n_test, maj, dtype=np.int64)
    proba = np.full(n_test, p_up, dtype=float)
    return pred, proba


def naive_prevday_baseline(log_return_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Same-day momentum rule: predict tomorrow's direction using today's realised return sign.

    Reference implementation: pred[t] = (log_return[t] > 0).
    """
    lr = np.asarray(log_return_test, dtype=float)
    pred = (lr > 0).astype(np.int64)
    proba = np.full_like(pred, 0.5, dtype=float)
    return pred, proba


def logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    grid = {"clf__C": list(config.logistic_grid["C"])}
    cv = TimeSeriesSplit(n_splits=config.inner_cv_splits)
    search = RandomizedSearchCV(
        pipe,
        grid,
        n_iter=len(grid["clf__C"]),
        cv=cv,
        scoring="roc_auc",
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    # Use threads so CV remains portable on macOS environments where loky semaphores are restricted.
    with parallel_backend("threading"):
        search.fit(X_train, y_train)
    y_pred = search.predict(X_test)
    y_proba = search.predict_proba(X_test)[:, 1]
    return y_pred.astype(np.int64), y_proba.astype(float), dict(search.best_params_)
