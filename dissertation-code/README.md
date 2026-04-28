# SPY Directional Forecasting — Creative Piece

Synoptic Project (6Z0019): Machine learning time-series forecasting for SPY with purged walk-forward validation and regime-stratified evaluation.

## Quick start

Requires **Python 3.13**. Verified on **Python 3.13.9**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
python -m src.main --quick
```

## Reproducibility and verification

This project is deterministic from the bundled dataset file `data/spy_raw.csv`.

The official submission artefacts are the outputs from the final full experimental run stored in `results/`, `figures/`, and `models/`. The `quick/` subdirectories are optional reproducibility artefacts for a lighter verification run.

After `python -m src.main --quick`, you should find:

- `results/quick/dataset_fingerprint.json` with a non-null `sha256`
- `results/quick/metrics_aggregate.csv`
- `figures/quick/walkforward_schematic.png`

Reproducibility checklist (clean machine):

- Use **Python 3.13**
- Install examiner dependencies with `pip install -e ".[dev]"`
- Run `pytest tests/` (should pass)
- Run `python -m src.main --quick`
- Verify the fingerprint file exists and points to the bundled CSV (do not re-download)

## XGBoost optional behaviour

XGBoost is a **nice-to-have**.

The default examiner install, `pip install -e ".[dev]"`, does not require XGBoost or SHAP. If `xgboost` cannot be imported on the examiner machine, the pipeline logs a warning and proceeds without XGBoost. In that case:

- Baselines (`majority`, `naive`, `logistic`) still run
- `rf` still runs
- There may be no `models/*/xgb_fold_*.pkl` files

For the full pinned research environment, use `pip install -r requirements.txt`. For optional extras without the pinned file, use `pip install -e ".[dev,research]"`.

## Two supported run modes

Use these as two distinct stories, not one muddled one:

- **Examiner quick path:** `pip install -e ".[dev]"` then `python -m src.main --quick`
- **Full dissertation reproduction path:** `pip install -e ".[dev,research]"` then `python -m src.main --require-research-deps`

The quick path is designed to stay runnable even if `xgboost` or `shap` are unavailable. The full dissertation path is stricter and will fail fast unless both are installed.

## Data cache

On first run, `src/data.py` downloads SPY via `yfinance` and saves **`data/spy_raw.csv`**. Record the download date here after generating the file used for submission:

**Data cached:** 2026-04-21 (real SPY from `yfinance`, 4,085 raw trading days, 2010-01-04 → 2026-03-31, auto-adjusted). After log-return and next-day target construction, the submitted fingerprint reports 4,083 labelled rows, 2010-01-05 → 2026-03-30.

Adjustments may change if `yfinance` recomputes splits/dividends; submitted results use the bundled CSV.

## Repository layout

- `config.yaml` — hyperparameters, date range, paths
- `src/` — pipeline modules (`data`, `features`, `validation`, `train`, …)
- `tests/` — pytest gates (target, features, walk-forward, baselines)
- `data/` — cached raw and engineered features
- `results/` — final submission CSVs, `run.log`, SHAP pickles
- `models/` — final submission pickled estimators per fold
- `figures/` — final submission 300 DPI PNGs and walk-forward animation
- `notebooks/pipeline.ipynb` — optional demo narration

## Examiner environment notes

- The project has been verified on a clean virtual environment with **Python 3.13.9**.
- If `xgboost` is not installed, the quick path still completes with baselines and Random Forest.
- If `ffmpeg` is unavailable, figure generation falls back from MP4 animation to GIF automatically.

## Security and EDI notes

This project is a local, offline-first research pipeline rather than a multi-user web application, so classic input-driven attacks such as SQL injection and XSS are not part of its normal threat surface. Even so, the project still has security and ethics considerations that should be acknowledged explicitly in the report and demo:

- **No personal data** is used. The dataset is public SPY market data, so the project does not process participant records or user-identifying information.
- **Untrusted model files are unsafe.** The submission includes pickle artefacts for reproducibility, but pickled files should not be treated as trusted external inputs because unpickling arbitrary files is a code-execution risk.
- **Dependency and environment risk** still exists. The project relies on third-party packages and optional tools such as `xgboost`, `shap`, and `ffmpeg`, so the examiner path is deliberately kept small and documented.
- **Network access is not required for normal verification.** The bundled `data/spy_raw.csv` is the frozen submission dataset; re-downloading is unnecessary and would weaken reproducibility.
- **EDI / bias limits still matter in finance ML.** Although the dataset contains no protected-characteristic columns, bias can still arise through the target definition, the market period sampled, class imbalance, regime imbalance, and the choice of features and baselines.
- **Claims should stay bounded.** The project forecasts next-day SPY direction under a leakage-safe protocol; it should not be framed as an objective or universally fair decision system.

## Project notes

- The feature set used in the submitted pipeline is **26** engineered features.
- The full run should be described consistently as the **final experimental run on the frozen dataset**.
- The headline result is that the more complex ML models did not outperform the long-only / majority baseline overall, even though some regime-level pockets were more promising.
- Security, ethics, and EDI considerations are summarized in `SECURITY_AND_EDI.md`.

## Video demonstration

Kaltura link: *(not included in this repo snapshot)*

## Submission checklist

Start with `INSTRUCTIONS.txt` for the fastest run path, then see `SUBMISSION_CHECKLIST.md` for package contents, verification commands, and supporting materials.
