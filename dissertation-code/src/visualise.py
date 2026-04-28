"""Figures for dissertation and demonstration (reads CSV/pickle outputs only)."""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

_CACHE_ROOT = Path(tempfile.gettempdir()) / "dissertation-code-cache"
_MPL_CACHE = _CACHE_ROOT / "matplotlib"
_XDG_CACHE = _CACHE_ROOT / "xdg"
_FONTCONFIG_CACHE = _XDG_CACHE / "fontconfig"
for path in (_CACHE_ROOT, _MPL_CACHE, _XDG_CACHE, _FONTCONFIG_CACHE):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from src.features import FEATURE_COLS

if TYPE_CHECKING:
    from src.config_loader import Config

logger = logging.getLogger(__name__)

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.prop_cycle": plt.cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]),
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def generate_all_figures(config: Config) -> None:
    """Generate Module 9 artefacts under figures_dir."""
    _apply_style()
    fig_dir = Path(config.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.results_dir)

    preds = pd.read_csv(results_dir / "predictions.csv", parse_dates=["date"])
    metrics = pd.read_csv(results_dir / "metrics.csv")
    agg = pd.read_csv(results_dir / "metrics_aggregate.csv")
    trade = pd.read_csv(results_dir / "trading_metrics.csv")
    regimes = pd.read_csv(results_dir / "regime_assignments.csv")

    # class distribution (from predictions y_true sample)
    plt.figure(figsize=(5, 4))
    # predictions.csv contains one row per (fold_id, model, date) so y_true is duplicated across models.
    # Use a single model's labels (or de-duplicate by fold_id+date) to avoid 5× inflated counts.
    if "model" in preds.columns and "rf" in set(preds["model"].astype(str).unique()):
        base = preds[preds["model"].astype(str) == "rf"]
    else:
        base = preds.drop_duplicates(subset=["fold_id", "date"])
    vc = base["y_true"].value_counts().sort_index()
    vc.plot(kind="bar", color=["#d62728", "#2ca02c"])
    plt.title("Target class distribution (test sets)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png")
    plt.close()

    # correlation heatmap — load feature cache if present
    feat_path = Path(config.feature_cache)
    if feat_path.exists():
        fd = pd.read_csv(feat_path, index_col=0, parse_dates=True)
        num = fd[FEATURE_COLS].corr()
        plt.figure(figsize=(12, 10))
        plt.imshow(num, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Feature correlation matrix")
        plt.tight_layout()
        plt.savefig(fig_dir / "correlation_heatmap.png")
        plt.close()

    # accuracy per fold
    acc = metrics[metrics["metric"] == "accuracy"]
    plt.figure(figsize=(10, 5))
    for model in sorted(acc["model"].unique()):
        sub = acc[acc["model"] == model].sort_values("fold_id")
        plt.plot(sub["fold_id"], sub["value"], marker="o", label=model, alpha=0.8)
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per fold")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_per_fold.png")
    plt.close()

    # metrics table image
    piv = agg.pivot_table(index="model", columns="metric", values="mean")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    tbl = ax.table(
        cellText=np.round(piv.values, 4),
        rowLabels=piv.index,
        colLabels=piv.columns,
        loc="center",
    )
    tbl.scale(1, 1.5)
    plt.savefig(fig_dir / "metrics_table.png")
    plt.close()

    # equity curves (from trading metrics simple return ranking)
    sub = trade[trade["cost_bps"] == 0]
    plt.figure(figsize=(8, 4))
    plt.bar(sub["model"], sub["cumulative_simple_return"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cumulative simple return")
    plt.title("Strategy cumulative simple return (0 bps)")
    plt.tight_layout()
    plt.savefig(fig_dir / "equity_curve.png")
    plt.close()

    # cost sensitivity small multiples
    costs = sorted(trade["cost_bps"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, c in zip(axes, costs):
        s = trade[trade["cost_bps"] == c]
        ax.bar(s["model"], s["cumulative_simple_return"])
        ax.set_title(f"{c} bps")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(fig_dir / "equity_curve_cost_sensitivity.png")
    plt.close()

    # drawdown placeholder from trading max_drawdown
    plt.figure(figsize=(8, 4))
    plt.bar(trade[trade["cost_bps"] == 0]["model"], trade[trade["cost_bps"] == 0]["max_drawdown"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Max drawdown")
    plt.tight_layout()
    plt.savefig(fig_dir / "drawdown_curve.png")
    plt.close()

    # confusion matrices aggregated (RF & XGB)
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, m in zip(axes, ["rf", "xgb"]):
        sub = preds[preds["model"] == m]
        if sub.empty:
            ax.axis("off")
            continue
        yt = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        cm = confusion_matrix(yt, yp, normalize="all")
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(m.upper())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Normalised confusion (aggregated test days)")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrices.png")
    plt.close()

    # walk-forward schematic static
    plt.figure(figsize=(10, 2))
    plt.barh([0], [5], color="C0", label="train")
    plt.barh([0], [0.2], left=[5], color="C3", label="purge")
    plt.barh([0], [1], left=[5.2], color="C2", label="test")
    plt.barh([0], [0.2], left=[6.2], color="0.7", label="embargo")
    plt.yticks([])
    plt.xlabel("Time (conceptual)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "walkforward_schematic.png")
    plt.close()

    # minimal animation (2 frames)
    fig, ax = plt.subplots(figsize=(8, 2))

    def _init() -> tuple:
        ax.clear()
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, 10)
        return ()

    def _frame(i: int) -> tuple:
        ax.clear()
        ax.barh([0], [5 + i], color="C0")
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(f"Walk-forward fold buildup ({i})")
        return ()

    anim = FuncAnimation(fig, _frame, frames=[0, 1], init_func=_init, blit=False)
    try:
        anim.save(fig_dir / "walkforward_animation.mp4", fps=30, dpi=150, writer="ffmpeg")
    except Exception:
        anim.save(fig_dir / "walkforward_animation.gif", fps=5, dpi=150, writer="pillow")
    plt.close(fig)

    # SHAP beeswarm from full SHAP pickles (one per trained representative fold)
    shap_dir = results_dir / "shap"
    if shap_dir.exists():
        try:
            for path in shap_dir.glob("*_full.pkl"):
                with open(path, "rb") as fh:
                    payload = pickle.load(fh)
                regime = str(payload.get("regime", "unknown"))
                model = str(payload.get("model", "rf"))
                sv = np.asarray(payload["shap_values"], dtype=float)
                if sv.ndim == 3:
                    sv = sv[..., 1]
                mean_abs = np.mean(np.abs(sv), axis=0).astype(float)
                order = np.asarray(np.argsort(mean_abs)[-15:], dtype=int).ravel()
                plt.figure(figsize=(8, 6))
                labels = [FEATURE_COLS[int(j)] for j in order][::-1]
                heights = mean_abs[order].ravel()[::-1]
                plt.barh(labels, heights, color="steelblue")
                plt.xlabel("Mean |SHAP|")
                plt.title(f"{model.upper()} SHAP (top 15) — {regime}")
                plt.tight_layout()
                plt.savefig(fig_dir / f"shap_{model}_{regime}.png", dpi=300)
                plt.close()
        except Exception as exc:
            logger.warning("SHAP figure generation failed: %s", exc)

    # Mean |SHAP| across folds (from *_summary.pkl) — replaces placeholder
    shap_summaries = []
    if shap_dir.exists():
        for p in shap_dir.glob("*_summary.pkl"):
            with open(p, "rb") as fh:
                payload = pickle.load(fh)
            shap_summaries.append(payload)

    if shap_summaries:
        s_df = pd.DataFrame(shap_summaries)
        # Also write a combined side-by-side figure to the canonical filename.
        models_in_sum = sorted(s_df["model"].unique())
        fig, axes = plt.subplots(1, len(models_in_sum), figsize=(6 * len(models_in_sum), 6), squeeze=False)
        for ax, model in zip(axes[0], models_in_sum):
            sub = s_df[s_df["model"] == model].copy()
            mat = np.vstack(sub["mean_abs_shap"].tolist())
            mean_abs = mat.mean(axis=0)
            top = np.argsort(mean_abs)[-15:].astype(int)
            labels = [FEATURE_COLS[int(i)] for i in top][::-1]
            heights = mean_abs[top][::-1]
            ax.barh(labels, heights, color="slateblue")
            ax.set_title(model.upper())
            ax.set_xlabel("Mean |SHAP| (fold-avg)")
        plt.suptitle("Feature importance (top 15) — mean |SHAP| across folds")
        plt.tight_layout()
        plt.savefig(fig_dir / "feature_importance.png", dpi=300)
        plt.close(fig)

        for model in sorted(s_df["model"].unique()):
            sub = s_df[s_df["model"] == model].copy()
            mat = np.vstack(sub["mean_abs_shap"].tolist())
            mean_abs = mat.mean(axis=0)
            top = np.argsort(mean_abs)[-15:].astype(int)
            labels = [FEATURE_COLS[int(i)] for i in top][::-1]
            heights = mean_abs[top][::-1]
            plt.figure(figsize=(8, 6))
            plt.barh(labels, heights, color="slateblue")
            plt.xlabel("Mean |SHAP| (fold-avg)")
            plt.title(f"Feature importance (top 15) — {model.upper()}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"feature_importance_{model}.png", dpi=300)
            plt.close()

    # Feature importance by regime (top 10) using full SHAP pickles (rep folds)
    if shap_dir.exists():
        rows = []
        for p in shap_dir.glob("*_full.pkl"):
            with open(p, "rb") as fh:
                payload = pickle.load(fh)
            regime = str(payload.get("regime", "unknown"))
            model = str(payload.get("model", "rf"))
            sv = np.asarray(payload["shap_values"], dtype=float)
            if sv.ndim == 3:
                sv = sv[..., 1]
            mean_abs = np.mean(np.abs(sv), axis=0)
            for i, val in enumerate(mean_abs.tolist()):
                rows.append({"model": model, "regime": regime, "feature": FEATURE_COLS[i], "mean_abs": val})

        if rows:
            r_df = pd.DataFrame(rows)
            # pick global top 10 per model by averaging across regimes
            panels = []
            for model in sorted(r_df["model"].unique()):
                mdf = r_df[r_df["model"] == model]
                top_feats = (
                    mdf.groupby("feature")["mean_abs"].mean().sort_values(ascending=False).head(10).index.tolist()
                )
                panels.append((model, top_feats))

            n = len(panels)
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
            for ax, (model, feats) in zip(axes[0], panels):
                pivot = (
                    r_df[(r_df["model"] == model) & (r_df["feature"].isin(feats))]
                    .pivot_table(index="feature", columns="regime", values="mean_abs", aggfunc="mean")
                    .reindex(feats)
                )
                pivot.plot(kind="barh", ax=ax)
                ax.set_title(model.upper())
                ax.set_xlabel("Mean |SHAP|")
            plt.suptitle("Feature importance by return regime (rep folds)")
            plt.tight_layout()
            plt.savefig(fig_dir / "feature_importance_by_regime.png", dpi=300)
            plt.close(fig)

    logger.info("Figures written to %s", fig_dir)
