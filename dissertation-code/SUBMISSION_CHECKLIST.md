# 6Z0019 Submission Checklist

Use this before creating the Creative Piece ZIP and final Report PDF.

## Creative Piece ZIP

- [ ] Include this project folder with `src/`, `tests/`, `config.yaml`, `pyproject.toml`, `requirements.txt`, `README.md`, and `INSTRUCTIONS.txt`.
- [ ] Include the frozen dataset at `data/spy_raw.csv`.
- [ ] Include the official final submission artefacts from the final full experimental run: `results/`, `figures/`, and `models/`.
- [ ] Include `results/dataset_fingerprint.json` from the submitted full run. This proves which CSV was used.
- [ ] `results/quick/`, `figures/quick/`, and `models/quick/` are intentionally empty — they are generated fresh when the examiner runs `python -m src.main --quick`.
- [ ] Name the ZIP using the required MMU pattern: `Lastname_Firstname_UID_CreativePiece.zip`.

## Before Submission

- [ ] Add the Kaltura/MMUTube share link for the 10-minute demo video in Moodle.
- [ ] Remove placeholder video/report links from any copied submission notes.

## Examiner Verification Path

From inside the unzipped `dissertation-code` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,research]"
python -m pytest tests/ -q
python -m src.main --quick
```

Notes:
- [ ] Use Python 3.13 for the examiner run path.
- [ ] Use `python -m pytest`, not bare `pytest`, to avoid PATH issues.
- [ ] The `results/quick/` and `figures/quick/` folders are empty on purpose — they get generated fresh by the run above.

Expected quick-run evidence:

- [ ] `results/quick/dataset_fingerprint.json`
- [ ] `results/quick/metrics_aggregate.csv`
- [ ] `results/quick/significance_tests.csv`
- [ ] `results/quick/trading_metrics.csv`
- [ ] `figures/quick/walkforward_schematic.png`
- [ ] `figures/quick/metrics_table.png`

## Report / documentation alignment

- [ ] Introduction and literature survey: explain the problem as leakage-safe SPY next-day direction forecasting, not a promise of profitable trading.
- [ ] Design and implementation: describe the pipeline stages and cite the config values exactly: 5-year train window, 126 trading-row test window, 200-row purge, 6-month plus 10-day calendar advance.
- [ ] Results and evaluation: use the generated full-run CSVs and figures in `results/` and `figures/` as the source of truth. Compare RF/XGBoost against majority, naive, and logistic baselines.
- [ ] Make the headline result explicit: the complex ML models did not outperform the long-only / majority baseline overall on the final experimental run.
- [ ] Do not describe the submitted full run as a rerun; describe it as the final experimental run on the frozen dataset.
- [ ] Include security discussion covering dependency trust, frozen-data reproducibility, and the fact that pickled model files should not be treated as trusted external inputs.
- [ ] Include EDI / ethics discussion covering the absence of personal data alongside remaining fairness and bias concerns from target definition, period selection, class imbalance, and scarce bear-regime coverage.
- [ ] Conclusions and academic quality: state limitations clearly, especially dataset non-stationarity, scarce bear-regime folds, optional XGBoost availability, multiple comparisons, and that pickled models should not be treated as trusted external inputs.
- [ ] Include the ethics declaration page and MMU Ethics Approval number in the Report.
- [ ] Append the Feasibility Study or Terms of Reference and the MMU EthOS approval email.
- [ ] Name the Report PDF using the required MMU pattern: `Lastname_Firstname_UID_Report.pdf`.

## Demo Video Structure

Use the 10 minutes like this:

- 1 min: aim and research challenge.
- 2 min: methodology, focusing on purged walk-forward validation and the frozen dataset hash.
- 3 min: live/product demonstration with `pytest tests/`, `python -m src.main --quick`, and the generated artefact folders.
- 2 min: key results and figures.
- 1 min: limitations and wider implications.
- 1 min: what you learned and follow-on work.
