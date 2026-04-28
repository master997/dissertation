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

## Reproducibility & verification (what to check)

This project is deterministic from the bundled dataset file `data/spy_raw.csv`.

The official submission artefacts are the outputs from the final full experimental run stored in `results/`, `figures/`, and `models/`. The `quick/` subdirectories are optional examiner-facing reproducibility artefacts only.

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

## Dissertation alignment (manual edits before submission)

- Update dissertation Table 3.2 to **26** features (not “approximately 30”).
- Update dissertation §1.2.2 objectives **1 and 3** to refer to **XGBoost specifically** (see Creative Piece spec §2).
- Describe the full run as the **final experimental run on the frozen dataset**, not as a rerun.
- Report the headline finding honestly: the more complex ML models did not outperform the long-only / majority baseline overall, even though some regime-level pockets were more promising.
- Add a short **security** subsection explaining why classic web/database attacks are not central here, but why dependency trust, external downloads, and pickle safety still matter.
- Add a short **EDI / ethics** subsection explaining that financial-market data avoids direct personal data, but does not remove bias risk from target construction, data period choice, class imbalance, and model interpretation.

## Video demonstration

Kaltura link: *(not included in this repo snapshot)*

## Submission checklist

Start with `INSTRUCTIONS.txt` for the fastest examiner run path, then see `SUBMISSION_CHECKLIST.md` before packaging. The checklist maps the Creative Piece ZIP contents, examiner run commands, and report/demo reminders to the 6Z0019 marking criteria.
