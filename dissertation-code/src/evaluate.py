"""
Evaluation: per-fold metrics, bootstrap CIs, significance, SHAP, regime splits.

References: Lundberg & Lee (2017); Lo (2004) for regime analysis.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import FEATURE_COLS
from src.stats import diebold_mariano_test, holm_corrected_ttests

if TYPE_CHECKING:
    from src.config_loader import Config

logger = logging.getLogger(__name__)

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore[assignment]


def _metric_row(fold_id: int, model: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> list[dict]:
    rows = []
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    try:
        ll = log_loss(y_true, y_proba)
    except ValueError:
        ll = float("nan")
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "log_loss": ll,
    }
    for k, v in m.items():
        rows.append({"fold_id": fold_id, "model": model, "metric": k, "value": float(v)})
    return rows


def compute_metrics(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read predictions.csv; write metrics.csv and metrics_aggregate.csv."""
    results_dir = Path(config.results_dir)
    pred_path = results_dir / "predictions.csv"
    preds = pd.read_csv(pred_path, parse_dates=["date"])

    long_rows: list[dict] = []
    for (fold_id, model), g in preds.groupby(["fold_id", "model"]):
        yt = g["y_true"].to_numpy(dtype=int)
        yp = g["y_pred"].to_numpy(dtype=int)
        pr = g["y_proba"].to_numpy(dtype=float)
        long_rows.extend(_metric_row(int(fold_id), str(model), yt, yp, pr))

    metrics_long = pd.DataFrame(long_rows)
    metrics_long.to_csv(results_dir / "metrics.csv", index=False)

    # Bootstrap CIs on fold-level metric values (1000 resamples)
    agg_rows: list[dict] = []
    rng = np.random.default_rng(config.seed)
    for model in sorted(preds["model"].unique()):
        for metric in METRICS:
            sub = metrics_long[(metrics_long["model"] == model) & (metrics_long["metric"] == metric)]
            vals = sub["value"].to_numpy(dtype=float)
            if len(vals) < 2:
                mean_v = float(np.mean(vals)) if len(vals) else float("nan")
                lo = hi = mean_v
            else:
                means = []
                for _ in range(1000):
                    sample = rng.choice(vals, size=len(vals), replace=True)
                    means.append(float(np.mean(sample)))
                mean_v = float(np.mean(vals))
                lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
            agg_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": mean_v,
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(results_dir / "metrics_aggregate.csv", index=False)
    return metrics_long, agg


def run_significance_tests(config: Config) -> pd.DataFrame:
    """Paired t-tests (Holm) on per-fold accuracy; Diebold-Mariano on Brier scores (pooled)."""
    results_dir = Path(config.results_dir)
    preds = pd.read_csv(results_dir / "predictions.csv", parse_dates=["date"])

    models = sorted(preds["model"].unique().astype(str))
    fold_ids = sorted(preds["fold_id"].unique())
    fold_acc: dict[str, list[float]] = {m: [] for m in models}
    for m in models:
        for fid in fold_ids:
            g = preds[(preds["fold_id"] == fid) & (preds["model"] == m)]
            if g.empty:
                fold_acc[m].append(float("nan"))
                continue
            yt = g["y_true"].to_numpy(dtype=float)
            yp = g["y_pred"].to_numpy(dtype=float)
            fold_acc[m].append(float(np.mean(yt == yp)))

    comparisons_all = [
        ("majority", "rf"),
        ("majority", "xgb"),
        ("logistic", "rf"),
        ("logistic", "xgb"),
        ("rf", "xgb"),
    ]
    available = set(models)
    comparisons = [(a, b) for (a, b) in comparisons_all if a in available and b in available]
    t_df = holm_corrected_ttests(fold_acc, comparisons)

    # DM on Brier: concatenate all test days per model pair
    dm_rows: list[dict] = []
    for a, b in comparisons:
        ga = preds[preds["model"] == a]
        gb = preds[preds["model"] == b]
        merged = ga.merge(
            gb,
            on=["fold_id", "date"],
            suffixes=("_a", "_b"),
        )
        if merged.empty:
            continue
        ya = merged["y_true_a"].to_numpy(dtype=float)
        pa = merged["y_proba_a"].to_numpy(dtype=float)
        pb = merged["y_proba_b"].to_numpy(dtype=float)
        loss_a = (pa - ya) ** 2
        loss_b = (pb - ya) ** 2
        dm, p = diebold_mariano_test(loss_a, loss_b, max_lag=5)
        dm_rows.append(
            {
                "comparison": f"{a}_vs_{b}",
                "test": "diebold_mariano",
                "statistic": dm,
                "p_value": p,
                "p_value_corrected": float("nan"),
            }
        )
    dm_df = pd.DataFrame(dm_rows)

    out = pd.concat([t_df, dm_df], ignore_index=True)
    out.to_csv(results_dir / "significance_tests.csv", index=False)
    return out


def regime_stratified_metrics(config: Config) -> pd.DataFrame:
    """Join metrics with regime_assignments; aggregate mean accuracy/auc per regime."""
    results_dir = Path(config.results_dir)
    regs = pd.read_csv(results_dir / "regime_assignments.csv")
    met = pd.read_csv(results_dir / "metrics.csv")
    acc = met[met["metric"] == "accuracy"][["fold_id", "model", "value"]].rename(
        columns={"value": "accuracy"}
    )
    auc = met[met["metric"] == "roc_auc"][["fold_id", "model", "value"]].rename(
        columns={"value": "roc_auc"}
    )
    m = acc.merge(auc, on=["fold_id", "model"])
    m = m.merge(regs[["fold_id", "return_regime", "vol_regime"]], on="fold_id", how="left")
    g = (
        m.groupby(["return_regime", "vol_regime", "model"])[["accuracy", "roc_auc"]]
        .mean()
        .reset_index()
    )
    g.to_csv(results_dir / "metrics_by_regime.csv", index=False)
    return g


def extract_shap(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: Config,
    regimes: pd.DataFrame,
    quick: bool,
) -> None:
    """Compute SHAP for Tree models; full detail for representative folds only."""
    if shap is None:
        logger.warning("SHAP is unavailable; skipping SHAP extraction.")
        return

    shap_dir = Path(config.results_dir) / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    fold_results_path = Path(config.results_dir) / "fold_results.csv"
    trained_ids: set[int] = set()
    if fold_results_path.exists():
        fr = pd.read_csv(fold_results_path)
        trained_ids = set(int(x) for x in fr["fold_id"].unique())

    # Representative fold ids: first *trained* fold in each return_regime
    rep: dict[str, int] = {}
    for r in ["bull", "bear", "sideways"]:
        sub = regimes[regimes["return_regime"] == r].sort_values("fold_id")
        if trained_ids:
            sub = sub[sub["fold_id"].isin(trained_ids)]
        if not sub.empty:
            rep[r] = int(sub.iloc[0]["fold_id"])

    models = ("rf", "xgb")

    fold_map = {i: (tr, te) for i, (tr, te) in enumerate(folds)}

    for regime, fold_id in rep.items():
        if fold_id not in fold_map:
            continue
        _, test_idx = fold_map[fold_id]
        X_test = df.iloc[test_idx][FEATURE_COLS].to_numpy(dtype=float)
        for m in models:
            path = Path(config.models_dir) / f"{m}_fold_{fold_id}.pkl"
            if not path.exists():
                logger.warning("Missing model for SHAP: %s", path)
                continue
            with open(path, "rb") as fh:
                est = pickle.load(fh)
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(X_test)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[..., 1]
            payload = {
                "shap_values": sv,
                "feature_names": FEATURE_COLS,
                "fold_id": fold_id,
                "model": m,
                "regime": regime,
            }
            out_p = shap_dir / f"{m}_fold_{fold_id}_full.pkl"
            with open(out_p, "wb") as fh:
                pickle.dump(payload, fh)

    # Compact: mean abs SHAP per fold not in rep (skip in quick minimal)
    if quick:
        return

    rep_set = set(rep.values())
    for fold_id, (_, test_idx) in enumerate(folds):
        if fold_id in rep_set:
            continue
        X_test = df.iloc[test_idx][FEATURE_COLS].to_numpy(dtype=float)
        for m in models:
            path = Path(config.models_dir) / f"{m}_fold_{fold_id}.pkl"
            if not path.exists():
                continue
            with open(path, "rb") as fh:
                est = pickle.load(fh)
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(X_test)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[..., 1]
            summary = np.mean(np.abs(sv), axis=0)
            compact = {"mean_abs_shap": summary.tolist(), "fold_id": fold_id, "model": m}
            with open(shap_dir / f"{m}_fold_{fold_id}_summary.pkl", "wb") as fh:
                pickle.dump(compact, fh)
