"""Feature definitions and fold-safe preprocessing shared by training and the API.

DATASET: UCI "Diabetes 130-US Hospitals for Years 1999-2008" (Strack et al., 2014) -
real, de-identified inpatient encounters from the Cerner Health Facts warehouse, CC BY 4.0.
Target: ``time_in_hospital`` (whole days, 1-14), renamed ``Length_of_Stay``.

Raw columns are mapped to readable names in ``training/data.py``. This module is
dependency-light (numpy/pandas only) so the API can reproduce preprocessing exactly.

LEAKAGE POLICY
--------------
Feature set A contains only information plausibly available near admission:
demographics, admission type/source, primary diagnosis group, the diabetes medication
regimen, HbA1c / glucose result ranges, and visit counts from the PRECEDING year.

Set B (research only, never served) adds variables that are accumulated DURING the stay or
recorded AT DISCHARGE - lab-procedure count, procedure count, medication count, number of
coded diagnoses, medication change, discharge disposition - plus payer and admitting
specialty. These are flagged as leakage risks; set B exists to show how much they inflate
apparent accuracy.

Identifiers (encounter_id, patient_nbr) are never predictors. Race is used only to check
error parity across groups, not as a predictor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TARGET = "Length_of_Stay"

# Never usable as predictors by any model.
FORBIDDEN_PREDICTORS = {TARGET, "encounter_id", "patient_nbr", "readmitted", "Race"}

# ---- Feature set A: admission-time -------------------------------------------------
SET_A_CATEGORICAL = ["Gender", "Admission Type", "Admission Source", "Medical Condition", "Medication",
                     "A1C Result", "Glucose Serum"]
SET_A_NUMERIC = ["Age", "Prior Inpatient", "Prior Emergency", "Prior Outpatient"]

# ---- Feature set B: expanded research features (leakage-prone) ---------------------
SET_B_EXTRA_CATEGORICAL = ["Discharge Disposition", "Medication Change", "Payer", "Medical Specialty"]
SET_B_EXTRA_NUMERIC = ["Lab Procedures", "Procedures", "Medications Count", "Diagnoses Count"]
LEAKAGE_RISK_FEATURES = {"Lab Procedures", "Procedures", "Medications Count", "Diagnoses Count",
                         "Discharge Disposition", "Medication Change"}

# Inner-loop feature-selection options for set A (chosen by Optuna inside inner CV).
FEATURE_SUBSETS_A = {
    "A_full": SET_A_CATEGORICAL + SET_A_NUMERIC,
    "A_no_prior_visits": SET_A_CATEGORICAL + ["Age"],
    "A_no_lab_results": [c for c in SET_A_CATEGORICAL if c not in ("A1C Result", "Glucose Serum")] + SET_A_NUMERIC,
}

CATEGORY_LEVELS = {
    "Gender": ["Female", "Male"],
    "Admission Type": ["Elective", "Emergency", "Other/Unknown", "Urgent"],
    "Admission Source": ["Emergency Room", "Other/Unknown", "Referral", "Transfer"],
    "Medical Condition": ["Circulatory", "Diabetes", "Digestive", "Genitourinary", "Injury", "Musculoskeletal",
                          "Neoplasms", "Other", "Respiratory"],
    "Medication": ["Insulin + oral", "Insulin only", "None", "Oral agents only"],
    "A1C Result": [">7%", ">8%", "Normal", "Not tested"],
    "Glucose Serum": [">200", ">300", "Normal", "Not tested"],
}
NUMERIC_RANGES = {"Age": (5, 95), "Prior Inpatient": (0, 10), "Prior Emergency": (0, 10), "Prior Outpatient": (0, 10)}


def feature_columns(feature_set: str, subset: str = "A_full") -> tuple[list[str], list[str]]:
    """Return (categorical, numeric) column lists for a feature set / subset."""
    if feature_set == "A":
        cols = FEATURE_SUBSETS_A[subset]
        cats = [c for c in cols if c in SET_A_CATEGORICAL]
        nums = [c for c in cols if c in SET_A_NUMERIC]
    elif feature_set == "B":
        cats = SET_A_CATEGORICAL + SET_B_EXTRA_CATEGORICAL
        nums = SET_A_NUMERIC + SET_B_EXTRA_NUMERIC
    else:
        raise ValueError(f"unknown feature set {feature_set!r}")
    bad = FORBIDDEN_PREDICTORS.intersection(cats + nums)
    assert not bad, f"forbidden predictors requested: {bad}"
    return cats, nums


def prepare_frame(df: pd.DataFrame, feature_set: str, encoder=None) -> pd.DataFrame:
    """All derived columns are built row-wise in the loader, so no fold-fitted state is needed."""
    return df


@dataclass
class TabularEncoder:
    """Fold-fitted encoder producing numeric matrices.

    * categoricals -> integer codes (0 = unknown) for embeddings, or one-hot columns
    * numerics -> standardized with training-fold mean / std
    """
    categorical: list[str]
    numeric: list[str]
    levels: dict[str, list[str]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        self.levels = {c: sorted(df[c].astype(str).unique().tolist()) for c in self.categorical}
        for c in self.numeric:
            v = df[c].astype(float)
            self.means[c] = float(v.mean())
            s = float(v.std(ddof=0))
            self.stds[c] = s if s > 1e-12 else 1.0
        return self

    def codes(self, df: pd.DataFrame) -> np.ndarray:
        if not self.categorical:
            return np.zeros((len(df), 0), dtype=np.int64)
        mats = []
        for c in self.categorical:
            lookup = {v: i + 1 for i, v in enumerate(self.levels[c])}
            mats.append(df[c].astype(str).map(lookup).fillna(0).astype(np.int64).to_numpy())
        return np.stack(mats, axis=1)

    def numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        if not self.numeric:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.stack([((df[c].astype(float) - self.means[c]) / self.stds[c]).to_numpy()
                         for c in self.numeric], axis=1).astype(np.float32)

    def one_hot_names(self) -> list[str]:
        names = []
        for c in self.categorical:
            names += [f"{c}={v}" for v in self.levels[c]]
        return names + list(self.numeric)

    def one_hot(self, df: pd.DataFrame, raw_numeric: bool = True) -> np.ndarray:
        blocks = []
        for c in self.categorical:
            vals = df[c].astype(str).to_numpy()
            lv = np.array(self.levels[c])
            blocks.append((vals[:, None] == lv[None, :]).astype(np.float32))
        if self.numeric:
            if raw_numeric:
                blocks.append(df[self.numeric].astype(float).to_numpy(dtype=np.float32))
            else:
                blocks.append(self.numeric_matrix(df))
        return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0), np.float32)

    def cardinalities(self) -> list[int]:
        return [len(self.levels[c]) + 1 for c in self.categorical]

    def to_dict(self) -> dict[str, Any]:
        return {"categorical": self.categorical, "numeric": self.numeric, "levels": self.levels,
                "means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TabularEncoder":
        return cls(categorical=d["categorical"], numeric=d["numeric"], levels=d["levels"],
                   means=d["means"], stds=d["stds"])


def age_group(age: pd.Series | np.ndarray) -> pd.Series:
    bins = [-np.inf, 39, 59, 69, 79, np.inf]
    labels = ["<40", "40-59", "60-69", "70-79", "80+"]
    return pd.cut(pd.Series(age).astype(float), bins=bins, labels=labels).astype(str)
