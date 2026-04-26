# SPY Directional Forecasting — Creative Piece

Synoptic Project (6Z0019): Machine learning time-series forecasting for SPY with purged walk-forward validation and regime-stratified evaluation.

## Quick start

See `INSTRUCTIONS.txt`. Requires **Python 3.11**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/
python -m src.main --quick
```

## Reproducibility & verification (what to check)

This project is deterministic from the bundled dataset file `data/spy_raw.csv`.

After `python -m src.main --quick`, you should find:

- `results/quick/dataset_fingerprint.json` with a non-null `sha256`
- `results/quick/metrics_aggregate.csv`
- `figures/quick/walkforward_schematic.png`

Reproducibility checklist (clean machine):

- Use **Python 3.11**
- Install dependencies from `requirements.txt`
- Run `pytest tests/` (should pass)
- Run `python -m src.main --quick`
- Verify the fingerprint file exists and points to the bundled CSV (do not re-download)

## XGBoost optional behaviour

XGBoost is a **nice-to-have**.

If `xgboost` cannot be imported on the examiner machine, the pipeline logs a warning and proceeds without XGBoost. In that case:

- Baselines (`majority`, `naive`, `logistic`) still run
- `rf` still runs
- There may be no `models/*/xgb_fold_*.pkl` files

## Data cache

On first run, `src/data.py` downloads SPY via `yfinance` and saves **`data/spy_raw.csv`**. Record the download date here after generating the file used for submission:

**Data cached:** 2026-04-21 (real SPY from `yfinance`, 4,085 trading days, 2010-01-04 → 2026-03-31, auto-adjusted).

Adjustments may change if `yfinance` recomputes splits/dividends; submitted results use the bundled CSV.

## Repository layout

- `config.yaml` — hyperparameters, date range, paths
- `src/` — pipeline modules (`data`, `features`, `validation`, `train`, …)
- `tests/` — pytest gates (target, features, walk-forward, baselines)
- `data/` — cached raw and engineered features
- `results/` — CSVs, `run.log`, SHAP pickles
- `models/` — pickled estimators per fold
- `figures/` — 300 DPI PNGs and walk-forward animation
- `notebooks/pipeline.ipynb` — optional demo narration

## Dissertation alignment (manual edits before submission)

- Update dissertation Table 3.2 to **26** features (not “approximately 30”).
- Update dissertation §1.2.2 objectives **1 and 3** to refer to **XGBoost specifically** (see Creative Piece spec §2).

## Video demonstration

Kaltura link: *(not included in this repo snapshot)*
