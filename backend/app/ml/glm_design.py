"""Design-matrix construction for the Gamma GLM (shared by training and inference)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import age_group

AGE_CENTER = 50.0
AGE_SCALE = 10.0
AGE_GROUPS = ["<30", "30-44", "45-59", "60-74", "75+"]


@dataclass
class GLMDesign:
    """Treatment-coded design: intercept + (levels-1) dummies per factor + an age term.

    Reference levels are the first (alphabetical) level seen in the training fold.
    """
    categorical: list[str]
    age_term: str = "linear"  # linear | quadratic | binned
    levels: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "GLMDesign":
        self.levels = {c: sorted(df[c].astype(str).unique().tolist()) for c in self.categorical}
        return self

    def column_names(self) -> list[str]:
        names = ["Intercept"]
        for c in self.categorical:
            names += [f"{c}[{v}]" for v in self.levels[c][1:]]
        if self.age_term == "linear":
            names += ["Age (per 10y, centred 50)"]
        elif self.age_term == "quadratic":
            names += ["Age (per 10y, centred 50)", "Age^2"]
        elif self.age_term == "binned":
            names += [f"Age group[{g}]" for g in AGE_GROUPS[1:]]
        return names

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        cols = [np.ones(n)]
        for c in self.categorical:
            v = df[c].astype(str).to_numpy()
            for lvl in self.levels[c][1:]:
                cols.append((v == lvl).astype(float))
        a = (df["Age"].astype(float).to_numpy() - AGE_CENTER) / AGE_SCALE
        if self.age_term == "linear":
            cols.append(a)
        elif self.age_term == "quadratic":
            cols += [a, a ** 2]
        elif self.age_term == "binned":
            g = age_group(df["Age"]).to_numpy()
            for lvl in AGE_GROUPS[1:]:
                cols.append((g == lvl).astype(float))
        return np.column_stack(cols)

    def to_dict(self) -> dict[str, Any]:
        return {"categorical": self.categorical, "age_term": self.age_term, "levels": self.levels}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GLMDesign":
        return cls(categorical=d["categorical"], age_term=d["age_term"], levels=d["levels"])
