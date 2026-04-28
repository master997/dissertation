# Security and EDI Notes

This note exists to help align the Creative Piece with the module guidance on security, equality, diversity, and inclusion.

## Security

This project is not a web application and does not expose forms, authentication, SQL queries, or user-authored HTML. That means the usual lecture examples such as SQL injection and XSS are not the main risks here.

The more realistic security considerations are:

- The project depends on third-party Python packages, so dependency trust and environment control matter.
- The submitted dataset is frozen in `data/spy_raw.csv`; examiner verification should use the bundled file rather than downloading fresh data.
- Pickled model artefacts in `models/` should not be treated as trusted external inputs, because Python pickle can execute code during loading.
- The documented quick-run path is intentionally small to reduce examiner friction and configuration risk.

Practical mitigation already present in the repo:

- Terminal-based execution path documented in `README.md`, `RUN_ME.txt`, and `SUBMISSION_CHECKLIST.md`.
- Automated tests via `pytest tests/`.
- Bundled frozen dataset for reproducibility.
- Graceful degradation when optional dependencies such as XGBoost or SHAP are unavailable.

## Equality, Diversity and Inclusion

The project does not use personal, demographic, or participant data. That removes some direct discrimination risks, but it does not remove fairness concerns altogether.

Relevant EDI considerations for this project:

- The target definition is a modelling choice and can privilege one framing of market behaviour over others.
- The sampled date range may over-represent some market regimes and under-represent others.
- Class balance and regime balance affect how reliable the learned patterns are across different conditions.
- Feature choices can still encode narrow assumptions about what matters in market prediction.

How this project addresses that responsibly:

- It compares ML models against simple baselines instead of assuming complexity is better.
- It reports limitations explicitly, including scarce bear-regime folds.
- It avoids over-claiming that the system is universally reliable or suitable for high-stakes automated decision-making.

## Positioning

This Creative Piece does not process personal data or provide a public-facing web interface, so classic application-security issues such as SQL injection and XSS are not central risks. The more relevant concerns are reproducibility, dependency trust, and the unsafe nature of untrusted pickle artefacts.

From an EDI perspective, the dataset contains no protected-characteristic fields, but this does not eliminate bias risk entirely: modelling choices, class imbalance, and uneven market-regime coverage can still shape the conclusions. The findings should therefore be read as limited methodological evidence rather than neutral or universally applicable truth.
