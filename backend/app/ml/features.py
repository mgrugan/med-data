"""Feature definitions and fold-safe preprocessing shared by training and the API.

Everything in this module is deliberately dependency-light (numpy/pandas only) so the
deployed API can reproduce the exact preprocessing used during training without
shipping scikit-learn objects.

LEAKAGE POLICY
--------------
The admission-time model must never see information that only exists after discharge.
* ``Discharge Date`` and anything derived from it (including ``Length_of_Stay``) is the
  target, never a predictor.
* ``Billing Amount`` is a *final* bill in this dataset and is therefore unavailable at
  admission. It appears only in the research feature set B, explicitly flagged as a
  leakage-risk variable, so we can show what happens when it is (wrongly) included.
* ``Name`` is a direct identifier and is never used.
* ``Doctor`` and ``Hospital`` are Faker-generated strings that are nearly unique per row
  (~40k distinct values in ~55k rows). They behave like identifiers, not clinical
  factors, so they are excluded from set A. Set B only uses fold-fitted *frequency*
  encodings of them to test that claim empirically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TARGET = "Length_of_Stay"

RAW_COLUMNS = [
    "Name", "Age", "Gender", "Blood Type", "Medical Condition", "Date of Admission",
    "Doctor", "Hospital", "Insurance Provider", "Billing Amount", "Room Number",
    "Admission Type", "Discharge Date", "Medication", "Test Results",
]

# Columns that may NEVER be used as predictors by any model.
FORBIDDEN_PREDICTORS = {"Name", "Discharge Date", TARGET, "Date of Admission"}

# ---- Feature set A: clinically plausible, admission-time --------------------------
SET_A_CATEGORICAL = ["Gender", "Blood Type", "Medical Condition", "Admission Type",
                     "Medication", "Test Results"]
SET_A_NUMERIC = ["Age"]

# ---- Feature set B: expanded research features (experimental only) ----------------
SET_B_EXTRA_CATEGORICAL = ["Insurance Provider"]
SET_B_EXTRA_NUMERIC = ["Room Number", "admit_month", "admit_dow", "admit_year",
                       "Billing Amount", "doctor_freq", "hospital_freq"]
LEAKAGE_RISK_FEATURES = {"Billing Amount"}

# Inner-loop feature-selection options for set A (chosen by Optuna inside inner CV).
FEATURE_SUBSETS_A = {
    "A_full": SET_A_CATEGORICAL + SET_A_NUMERIC,
    "A_no_blood_type": [c for c in SET_A_CATEGORICAL if c != "Blood Type"] + SET_A_NUMERIC,
    "A_no_test_result": [c for c in SET_A_CATEGORICAL if c != "Test Results"] + SET_A_NUMERIC,
}

CATEGORY_LEVELS = {
    "Gender": ["Female", "Male"],
    "Blood Type": ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"],
    "Medical Condition": ["Arthritis", "Asthma", "Cancer", "Diabetes", "Hypertension", "Obesity"],
    "Admission Type": ["Elective", "Emergency", "Urgent"],
    "Medication": ["Aspirin", "Ibuprofen", "Lipitor", "Paracetamol", "Penicillin"],
    "Test Results": ["Abnormal", "Inconclusive", "Normal"],
    "Insurance Provider": ["Aetna", "Blue Cross", "Cigna", "Medicare", "UnitedHealthcare"],
}


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


def add_admission_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived from the admission date only (known at admission)."""
    out = df.copy()
    adm = pd.to_datetime(out["Date of Admission"])
    out["admit_month"] = adm.dt.month.astype(float)
    out["admit_dow"] = adm.dt.dayofweek.astype(float)
    out["admit_year"] = adm.dt.year.astype(float)
    return out


# ------------------------------------------------------------------------------------
# Fold-fitted encoders. Each has fit/transform and to_dict/from_dict.
# ------------------------------------------------------------------------------------
@dataclass
class FrequencyEncoder:
    """Maps a high-cardinality identifier to its count in the *training fold*."""
    columns: dict[str, str] = field(default_factory=lambda: {"Doctor": "doctor_freq",
                                                              "Hospital": "hospital_freq"})
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "FrequencyEncoder":
        self.counts = {c: df[c].value_counts().to_dict() for c in self.columns if c in df}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for src, dst in self.columns.items():
            if src in out:
                out[dst] = out[src].map(self.counts.get(src, {})).fillna(0).astype(float)
            else:
                out[dst] = 0.0
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "counts": self.counts}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrequencyEncoder":
        return cls(columns=d["columns"], counts=d["counts"])


def prepare_frame(df: pd.DataFrame, feature_set: str, freq_encoder: FrequencyEncoder | None) -> pd.DataFrame:
    """Add derived (admission-time) columns needed by the feature set."""
    if feature_set == "B":
        df = add_admission_calendar(df)
        assert freq_encoder is not None
        df = freq_encoder.transform(df)
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
    bins = [-np.inf, 29, 44, 59, 74, np.inf]
    labels = ["<30", "30-44", "45-59", "60-74", "75+"]
    return pd.cut(pd.Series(age).astype(float), bins=bins, labels=labels).astype(str)
