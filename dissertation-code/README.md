# SPY Directional Forecasting — Creative Piece

Synoptic Project (6Z0019) | Mohammed Master | 22476494

Machine learning time-series forecasting for SPY with purged walk-forward validation and regime-stratified evaluation.

---

## Start here

**Read `INSTRUCTIONS.txt` first.** It has the exact commands to run the project from scratch in 7 steps.

Short version:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,research]"
python -m pytest tests/ -q
python -m src.main --quick
```

Requires **Python 3.13**. Verified on **Python 3.13.9**.

---

## What the quick run produces

Running `python -m src.main --quick` generates everything fresh on your machine:

| File | What it is |
|---|---|
| `results/quick/dataset_fingerprint.json` | SHA-256 hash of the data file used |
| `results/quick/metrics_aggregate.csv` | Accuracy, F1, AUC per model across folds |
| `results/quick/significance_tests.csv` | Paired-t (Holm) and Diebold-Mariano p-values |
| `results/quick/trading_metrics.csv` | Sharpe ratio, drawdown at different cost levels |
| `figures/quick/walkforward_schematic.png` | Diagram of the 19-fold purged walk-forward scheme |
| `figures/quick/metrics_table.png` | Summary table comparing all models vs baselines |

---

## Full submission artefacts

The official final-run outputs are in `results/`, `figures/`, and `models/`. These came from the full 19-fold run on the frozen dataset `data/spy_raw.csv`. The quick run above is a reproducible subset — it does not overwrite these.

---

## Repository layout

```
src/           14-module Python pipeline
tests/         pytest leakage and correctness gates
data/          frozen dataset (spy_raw.csv) and feature cache
results/       final-run CSVs, SHAP pickles, run log
figures/       final-run 300 DPI PNGs
models/        final-run pickled estimators (RF and XGBoost per fold)
config.yaml    all hyperparameters and paths
```

---

## Key design decisions

- **Purged walk-forward CV** — 19 expanding folds, 5-year train, 6-month test, 200-row purge gap, 10-day embargo. Prevents data leakage across time (Lopez de Prado 2018, ch.7).
- **26 engineered features** — momentum, trend, volatility, volume families. All computed on past data only inside each training window.
- **Regime stratification** — predictions broken down into 9 cells (bull/bear/sideways x low/mid/high vol).
- **Statistical tests** — paired-t with Holm correction and Diebold-Mariano with HAC-robust variance.
- **Deterministic** — fixed seeds, SHA-256 dataset fingerprint, pytest gates fail the build if leakage is reintroduced.

---

## Headline result

Random Forest and XGBoost both scored 49.7% directional accuracy against a majority baseline of 55.4%. The gap is statistically significant (Holm-corrected p = 0.011, DM p < 10^-5). The complex models did not beat predicting "up" every day once leakage controls were applied.

---

## Security and EDI

See `SECURITY_AND_EDI.md`. Short version: no personal data used, pickle files should not be treated as trusted external inputs, reproducibility relies on the bundled CSV not re-downloading.
