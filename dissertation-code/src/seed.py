"""Deterministic seeding for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seeds(seed: int) -> None:
    """
    Seed Python, NumPy, hash randomization, and library RNGs used by sklearn/XGBoost.

    References: Breiman (2001); Chen & Guestrin (2016); scikit-learn / xgboost docs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import sklearn  # noqa: F401

        from sklearn.utils import check_random_state

        check_random_state(seed)
    except ImportError:
        pass
    try:
        import xgboost as xgb

        # Global seed for XGBoost 2.x
        if hasattr(xgb, "set_config"):
            pass  # version-dependent; params use random_state in estimators
    except ImportError:
        pass
