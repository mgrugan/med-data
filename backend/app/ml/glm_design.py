"""Design-matrix construction for the Gamma GLM (shared by training and inference)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import age_group

AGE_CENTER = 60.0
AGE_SCALE = 10.0
AGE_GROUPS = ["<40", "40-59", "60-69", "70-79", "80+"]


@dataclass
class GLMDesign:
    """Treatment-coded design: intercept + (levels-1) dummies per factor + an age term
    + log1p-transformed count variables (prior visits; set-B counts).

    Reference levels are the first (alphabetical) level seen in the training fold.
    """
    categorical: list[str]
    age_term: str = "linear"  # linear | quadratic | binned
    counts: list[str] = field(default_factory=list)
    levels: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "GLMDesign":
        self.levels = {c: sorted(df[c].astype(str).unique().tolist()) for c in self.categorical}
        return self

    def _columns(self) -> list[tuple[str, str]]:
        """(column name, source feature) for every design column."""
        cols = [("Intercept", "Intercept")]
        for c in self.categorical:
            cols += [(f"{c}[{v}]", c) for v in self.levels[c][1:]]
        if self.age_term == "linear":
            cols += [("Age (per 10y, centred 60)", "Age")]
        elif self.age_term == "quadratic":
            cols += [("Age (per 10y, centred 60)", "Age"), ("Age^2", "Age")]
        elif self.age_term == "binned":
            cols += [(f"Age group[{g}]", "Age") for g in AGE_GROUPS[1:]]
        cols += [(f"log1p({c})", c) for c in self.counts]
        return cols

    def column_names(self) -> list[str]:
        return [n for n, _ in self._columns()]

    def column_features(self) -> list[str]:
        return [f for _, f in self._columns()]

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
        for c in self.counts:
            cols.append(np.log1p(np.clip(df[c].astype(float).to_numpy(), 0, None)))
        return np.column_stack(cols)

    def to_dict(self) -> dict[str, Any]:
        return {"categorical": self.categorical, "age_term": self.age_term, "counts": self.counts,
                "levels": self.levels}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GLMDesign":
        return cls(categorical=d["categorical"], age_term=d["age_term"], counts=d.get("counts", []),
                   levels=d["levels"])
