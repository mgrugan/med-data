"""Data loading, target construction, de-duplication and the untouched holdout split."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.ml.features import TARGET
from .config import DATA_PATH, GLOBAL_SEED, HOLDOUT_FRACTION, SPLIT_DIR, CALIBRATION_FRACTION


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
    # Target: Length_of_Stay = Discharge Date - Date of Admission (days)
    df[TARGET] = (df["Discharge Date"] - df["Date of Admission"]).dt.days.astype(float)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate records.

    Exact duplicates would otherwise land on both sides of a split and leak test rows into
    training. Name casing is randomised in the source file, so names are normalised before
    comparing.
    """
    report = {"rows_raw": int(len(df))}
    key = df.assign(Name=df["Name"].str.lower().str.strip())
    dup = key.duplicated(keep="first")
    df = df.loc[~dup].reset_index(drop=True)
    report["exact_duplicates_removed"] = int(dup.sum())
    report["rows_clean"] = int(len(df))
    report["invalid_los_removed"] = int((df[TARGET] < 0).sum())
    df = df.loc[df[TARGET] >= 0].reset_index(drop=True)
    return df, report


@dataclass
class Splits:
    dev: pd.DataFrame        # development data (all CV, tuning, preprocessing fitting)
    test: pd.DataFrame       # untouched final holdout
    dev_fit_idx: np.ndarray  # rows of dev used to fit the final models
    dev_cal_idx: np.ndarray  # rows of dev used ONLY for conformal calibration


def _los_strata(y: pd.Series) -> pd.Series:
    return pd.qcut(y.rank(method="first"), q=10, labels=False)


def make_splits(df: pd.DataFrame, seed: int = GLOBAL_SEED, persist: bool = True) -> Splits:
    idx = np.arange(len(df))
    dev_idx, test_idx = train_test_split(idx, test_size=HOLDOUT_FRACTION, random_state=seed,
                                         stratify=_los_strata(df[TARGET]))
    dev = df.iloc[np.sort(dev_idx)].reset_index(drop=True)
    test = df.iloc[np.sort(test_idx)].reset_index(drop=True)
    fit_idx, cal_idx = train_test_split(np.arange(len(dev)), test_size=CALIBRATION_FRACTION,
                                        random_state=seed + 1, stratify=_los_strata(dev[TARGET]))
    if persist:
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / "split_manifest.json").write_text(json.dumps({
            "seed": seed, "holdout_fraction": HOLDOUT_FRACTION,
            "calibration_fraction_of_dev": CALIBRATION_FRACTION,
            "n_dev": int(len(dev)), "n_test": int(len(test)),
            "n_dev_fit": int(len(fit_idx)), "n_dev_calibration": int(len(cal_idx)),
            "stratification": "deciles of Length_of_Stay",
        }, indent=2))
        np.save(SPLIT_DIR / "test_original_index.npy", np.sort(test_idx))
    return Splits(dev=dev, test=test, dev_fit_idx=np.sort(fit_idx), dev_cal_idx=np.sort(cal_idx))


def describe(df: pd.DataFrame) -> dict:
    y = df[TARGET]
    hist = y.value_counts().sort_index()
    return {
        "n": int(len(df)),
        "los_mean": float(y.mean()), "los_median": float(y.median()), "los_sd": float(y.std()),
        "los_min": float(y.min()), "los_max": float(y.max()),
        "los_skew": float(y.skew()),
        "los_histogram": {str(int(k)): int(v) for k, v in hist.items()},
        "unique_doctors": int(df["Doctor"].nunique()), "unique_hospitals": int(df["Hospital"].nunique()),
        "unique_rooms": int(df["Room Number"].nunique()),
        "admission_range": [str(df["Date of Admission"].min().date()), str(df["Date of Admission"].max().date())],
        "group_means": {c: df.groupby(c)[TARGET].mean().round(3).to_dict()
                        for c in ["Gender", "Blood Type", "Medical Condition", "Admission Type",
                                  "Medication", "Test Results", "Insurance Provider"]},
        "correlations_with_los": {c: float(np.corrcoef(df[c].astype(float), y)[0, 1])
                                  for c in ["Age", "Billing Amount", "Room Number"]},
    }
