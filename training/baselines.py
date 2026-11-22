"""Naive baselines that every advanced model must beat."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import LOSModel


class MedianBaseline(LOSModel):
    """Baseline 1: predict the training-fold median LOS for everyone."""
    key = "median_baseline"
    display_name = "Median Baseline"
    family = "baseline"
    complexity = 0

    def fit(self, df, params=None):
        self.params = params or {}
        self.median_ = float(np.median(self.y(df)))
        return self

    def predict(self, df):
        return np.full(len(df), self.median_)


class ConditionMedianBaseline(LOSModel):
    """Baseline 2: median LOS within Medical Condition (training-fold medians)."""
    key = "condition_median"
    display_name = "Condition Median"
    family = "baseline"
    complexity = 1

    def fit(self, df, params=None):
        self.params = params or {}
        self.global_ = float(np.median(self.y(df)))
        self.table_ = df.groupby("Medical Condition")["Length_of_Stay"].median().to_dict()
        return self

    def predict(self, df):
        return df["Medical Condition"].map(self.table_).fillna(self.global_).to_numpy(dtype=float)
