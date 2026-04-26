"""
Purged walk-forward validation with embargo.

Reference: López de Prado (2018) ch. 7 — Advances in Financial Machine Learning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.config_loader import Config

logger = logging.getLogger(__name__)

# Approximate trading days per month for lookahead (embargo stepping uses calendar + explicit test length)
_TEST_DAYS = 126


def walk_forward_splits(df: pd.DataFrame, config: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Rolling-window walk-forward splits with purge gap and embargo between folds.

    - Train: `train_years` calendar years immediately preceding test_start (by date).
    - Purge: drop last `purge_days` trading rows from the raw train window.
    - Test: next `126` trading rows from test_start (≈ 6 months).
    - Advance (embargo semantics): calendar time — `test_months` months + `embargo_days` days
      from the *current fold's cursor*, then snap to the first available trading row on/after
      that timestamp.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    dates = df.index
    n = len(df)

    first_data = dates[0]
    # First fold: test starts after train_years from first observation
    test_start_cursor = first_data + pd.DateOffset(years=config.train_years)

    folds: list[tuple[np.ndarray, np.ndarray]] = []

    while True:
        # Test window: 126 trading rows starting at first row >= test_start_cursor
        test_mask = dates >= test_start_cursor
        if not test_mask.any():
            break
        test_start_pos = int(np.argmax(test_mask))
        test_end_pos = min(test_start_pos + _TEST_DAYS, n)
        if test_end_pos - test_start_pos < _TEST_DAYS:
            break  # not enough rows left for a full test window

        test_idx = np.arange(test_start_pos, test_end_pos, dtype=np.int64)
        test_first_date = dates[test_start_pos]

        # Train: 5 calendar years ending the trading day before test_first_date
        train_end_bound = test_first_date - pd.Timedelta(days=1)
        train_start_bound = train_end_bound - pd.DateOffset(years=config.train_years)

        train_mask = (dates >= train_start_bound) & (dates <= train_end_bound)
        train_positions = np.flatnonzero(np.asarray(train_mask, dtype=bool))

        if train_positions.size == 0:
            logger.warning("No training rows for fold starting %s — stopping.", test_first_date)
            break

        # Purge last purge_days rows from train
        if train_positions.size > config.purge_days:
            train_positions = train_positions[:-config.purge_days]
        else:
            logger.warning(
                "Train window too short after boundaries at %s — skipping fold.", test_first_date
            )
            test_start_cursor = test_start_cursor + pd.DateOffset(
                months=config.test_months, days=config.embargo_days
            )
            continue

        train_idx = train_positions.astype(np.int64)

        folds.append((train_idx, test_idx))

        logger.info(
            "Fold %d: train [%s .. %s] n=%d | test [%s .. %s] n=%d",
            len(folds) - 1,
            dates[train_idx[0]],
            dates[train_idx[-1]],
            len(train_idx),
            dates[test_idx[0]],
            dates[test_idx[-1]],
            len(test_idx),
        )

        # Next fold: advance test_start by test_months + embargo_days (calendar)
        test_start_cursor = test_start_cursor + pd.DateOffset(
            months=config.test_months, days=config.embargo_days
        )

        if test_start_cursor > dates[-1]:
            break

    return folds
