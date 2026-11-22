"""Central configuration: paths, seeds, and compute budgets.

All randomness in the project flows from ``GLOBAL_SEED`` through ``seed_everything`` and
explicit ``random_state`` arguments, so a full pipeline run is reproducible.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "healthcare_dataset.csv"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "validation" / "results"
FIG_DIR = ROOT / "validation" / "figures"

GLOBAL_SEED = 20240531
HOLDOUT_FRACTION = 0.20       # untouched final test set
CALIBRATION_FRACTION = 0.15   # of development data, reserved for split-conformal calibration
EARLY_STOP_FRACTION = 0.10    # carved from each *training* portion for early stopping
CONFORMAL_LEVELS = [0.5, 0.8, 0.9, 0.95]


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


@dataclass
class Budget:
    name: str
    outer_folds: int
    inner_folds: int
    inner_folds_nn: int
    cv_repeats: int
    trials: dict[str, int]
    permutations: dict[str, int]
    bootstrap: int
    learning_curve_seeds: int
    loocv_n: int
    subsample: int | None = None   # use only N rows (for smoke tests)
    ft_max_epochs: int = 40
    threads: int = 4
    models: list[str] = field(default_factory=lambda: [
        "median_baseline", "condition_median", "gamma_glm", "xgboost", "xgboost_aft",
        "catboost", "ft_transformer"])


BUDGETS = {
    # The configuration used for the committed results.
    "full": Budget(
        name="full", outer_folds=5, inner_folds=5, inner_folds_nn=3, cv_repeats=3,
        trials={"gamma_glm": 18, "xgboost": 25, "xgboost_aft": 15, "catboost": 12, "ft_transformer": 6},
        permutations={"median_baseline": 0, "condition_median": 200, "gamma_glm": 200,
                      "xgboost": 40, "xgboost_aft": 20, "catboost": 20, "ft_transformer": 5},
        bootstrap=1000, learning_curve_seeds=3, loocv_n=1000,
    ),
    # Fast end-to-end smoke test (used by the test suite / CI).
    "quick": Budget(
        name="quick", outer_folds=3, inner_folds=2, inner_folds_nn=2, cv_repeats=2,
        trials={"gamma_glm": 2, "xgboost": 2, "xgboost_aft": 2, "catboost": 2, "ft_transformer": 1},
        permutations={"median_baseline": 0, "condition_median": 5, "gamma_glm": 5,
                      "xgboost": 3, "xgboost_aft": 2, "catboost": 2, "ft_transformer": 1},
        bootstrap=200, learning_curve_seeds=1, loocv_n=100, subsample=4000, ft_max_epochs=4,
    ),
}
