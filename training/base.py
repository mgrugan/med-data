"""Common interface implemented by every model in the comparison.

Every model owns its preprocessing: ``fit`` receives only training-fold rows and fits all
encoders/scalers on them, so the validation fold never influences preprocessing.
Models that use early stopping carve an internal early-stopping subset out of the
*training* rows (never the validation/test fold).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.app.ml.features import TARGET, prepare_frame
from .config import EARLY_STOP_FRACTION, GLOBAL_SEED


def early_stop_split(n: int, seed: int, frac: float = EARLY_STOP_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = max(1, int(round(n * frac)))
    return np.sort(perm[k:]), np.sort(perm[:k])


class LOSModel:
    key: str = "base"
    display_name: str = "Base"
    family: str = "baseline"          # baseline | statistical | tree | neural
    complexity: int = 0               # used for parsimony in model selection
    likelihood_based: bool = False    # AIC/BIC are meaningful only if True
    uses_early_stopping: bool = False
    is_neural: bool = False

    def __init__(self, feature_set: str = "A", seed: int = GLOBAL_SEED, threads: int = 4, **kw: Any):
        self.feature_set = feature_set
        self.seed = seed
        self.threads = threads
        self.params: dict[str, Any] = {}
        self.freq_encoder = None  # no fold-fitted derived features for this dataset
        self.extra = kw

    # ---- hyper-parameters ---------------------------------------------------------
    def suggest(self, trial) -> dict[str, Any]:
        return {}

    def default_params(self) -> dict[str, Any]:
        return {}

    def grid(self) -> dict[str, list] | None:
        """Optional exhaustive grid (used with Optuna's GridSampler)."""
        return None

    # ---- preprocessing helpers -----------------------------------------------------
    def _prep_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_frame(df, self.feature_set, self.freq_encoder)

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_frame(df, self.feature_set, self.freq_encoder)

    # ---- API -----------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> "LOSModel":
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def clone(self) -> "LOSModel":
        return type(self)(feature_set=self.feature_set, seed=self.seed, threads=self.threads, **self.extra)

    @staticmethod
    def y(df: pd.DataFrame) -> np.ndarray:
        return df[TARGET].to_numpy(dtype=float)

    def best_iteration(self) -> int | None:
        return None
