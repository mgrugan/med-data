"""Data loading, cleaning, feature derivation and the untouched holdout split.

Source: UCI "Diabetes 130-US Hospitals for Years 1999-2008" (Strack et al., 2014,
BioMed Research International, doi:10.1155/2014/781670). CC BY 4.0. Real, de-identified
inpatient encounters (1-14 day stays) with a diabetes diagnosis from 130 US hospitals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.ml.features import TARGET
from .config import CALIBRATION_FRACTION, DATA_PATH, GLOBAL_SEED, HOLDOUT_FRACTION, SPLIT_DIR

DRUG_COLUMNS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
                "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
                "troglitazone", "tolazamide", "examide", "citoglipton", "glyburide-metformin", "glipizide-metformin",
                "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

# discharge_disposition_id values meaning death or hospice: the stay did not end in a discharge
# home/onward care, so these encounters are excluded from a "time until discharge" model.
DEATH_OR_HOSPICE = {11, 13, 14, 19, 20, 21}

ADMISSION_TYPE = {1: "Emergency", 2: "Urgent", 3: "Elective", 7: "Emergency"}          # 7 = trauma centre
ADMISSION_SOURCE = {7: "Emergency Room", 1: "Referral", 2: "Referral", 3: "Referral",
                    4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer", 22: "Transfer", 25: "Transfer"}
DISCHARGE = {1: "Home", 6: "Home with health service", 8: "Home with health service",
             3: "Skilled nursing / care facility", 4: "Skilled nursing / care facility",
             5: "Other inpatient facility", 2: "Other inpatient facility", 22: "Rehabilitation",
             23: "Other inpatient facility", 24: "Skilled nursing / care facility", 7: "Left against advice"}


def icd9_group(code) -> str:
    """Standard grouping of the primary ICD-9 diagnosis (as in Strack et al., 2014)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Other"
    s = str(code)
    if s.startswith(("V", "E")):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 250 <= v < 251:
        return "Diabetes"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def medication_regimen(df: pd.DataFrame) -> pd.Series:
    insulin = df["insulin"] != "No"
    oral = (df[DRUG_COLUMNS] != "No").any(axis=1)
    return pd.Series(np.select([insulin & oral, insulin, oral], ["Insulin + oral", "Insulin only", "Oral agents only"],
                               default="None"), index=df.index)


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path, na_values="?", low_memory=False)
    d = pd.DataFrame({
        "encounter_id": raw["encounter_id"], "patient_nbr": raw["patient_nbr"],
        # age is released in 10-year bands, e.g. "[70-80)" -> midpoint 75
        "Age": raw["age"].str.extract(r"\[(\d+)-")[0].astype(float) + 5,
        "Gender": raw["gender"],
        "Race": raw["race"].fillna("Unknown"),
        "Admission Type": raw["admission_type_id"].map(ADMISSION_TYPE).fillna("Other/Unknown"),
        "Admission Source": raw["admission_source_id"].map(ADMISSION_SOURCE).fillna("Other/Unknown"),
        "Medical Condition": raw["diag_1"].map(icd9_group),
        "Primary ICD-9": raw["diag_1"].fillna("Unknown").astype(str),
        "Medication": medication_regimen(raw),
        "A1C Result": raw["A1Cresult"].map({"Norm": "Normal", ">7": ">7%", ">8": ">8%"}).fillna("Not tested"),
        "Glucose Serum": raw["max_glu_serum"].map({"Norm": "Normal", ">200": ">200", ">300": ">300"}).fillna("Not tested"),
        "Prior Inpatient": raw["number_inpatient"].astype(float),
        "Prior Emergency": raw["number_emergency"].astype(float),
        "Prior Outpatient": raw["number_outpatient"].astype(float),
        # ---- set B (research only; accumulated during the stay or recorded at discharge)
        "Lab Procedures": raw["num_lab_procedures"].astype(float),
        "Procedures": raw["num_procedures"].astype(float),
        "Medications Count": raw["num_medications"].astype(float),
        "Diagnoses Count": raw["number_diagnoses"].astype(float),
        "Medication Change": raw["change"].map({"Ch": "Changed", "No": "No change"}),
        "Discharge Disposition": raw["discharge_disposition_id"].map(DISCHARGE).fillna("Other/Unknown"),
        "Payer": raw["payer_code"].fillna("Unknown"),
        "Medical Specialty": raw["medical_specialty"].fillna("Unknown"),
        "_discharge_id": raw["discharge_disposition_id"],
        TARGET: raw["time_in_hospital"].astype(float),
    })
    top = d["Medical Specialty"].value_counts().index[:12]
    d["Medical Specialty"] = d["Medical Specialty"].where(d["Medical Specialty"].isin(top), "Other")
    return d


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cohort definition.

    1. Keep each patient's FIRST encounter only. Repeat encounters of one patient are
       correlated; letting them fall on both sides of a split leaks patient-level information
       and violates the independence assumed by CV and the bootstrap.
    2. Exclude encounters ending in death or hospice (not a discharge).
    3. Exclude records with unknown/invalid gender (3 rows).
    """
    report = {"rows_raw": int(len(df)), "patients": int(df["patient_nbr"].nunique())}
    df = df.sort_values("encounter_id")
    first = ~df.duplicated("patient_nbr", keep="first")
    report["repeat_encounters_removed"] = int((~first).sum())
    df = df.loc[first]
    dead = df["_discharge_id"].isin(DEATH_OR_HOSPICE)
    report["death_or_hospice_removed"] = int(dead.sum())
    df = df.loc[~dead]
    bad = ~df["Gender"].isin(["Female", "Male"])
    report["invalid_gender_removed"] = int(bad.sum())
    df = df.loc[~bad].drop(columns=["_discharge_id"]).reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["exact_duplicates_removed"] = 0
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
            "dataset": "UCI Diabetes 130-US Hospitals 1999-2008", "seed": seed,
            "holdout_fraction": HOLDOUT_FRACTION, "calibration_fraction_of_dev": CALIBRATION_FRACTION,
            "n_dev": int(len(dev)), "n_test": int(len(test)),
            "n_dev_fit": int(len(fit_idx)), "n_dev_calibration": int(len(cal_idx)),
            "stratification": "deciles of Length_of_Stay", "unit": "first encounter per patient",
        }, indent=2))
        np.save(SPLIT_DIR / "test_encounter_ids.npy", test["encounter_id"].to_numpy())
    return Splits(dev=dev, test=test, dev_fit_idx=np.sort(fit_idx), dev_cal_idx=np.sort(cal_idx))


def describe(df: pd.DataFrame) -> dict:
    y = df[TARGET]
    hist = y.value_counts().sort_index()
    from backend.app.ml.features import SET_A_CATEGORICAL, SET_A_NUMERIC, SET_B_EXTRA_NUMERIC
    return {
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008 (real, de-identified; CC BY 4.0)",
        "n": int(len(df)),
        "los_mean": float(y.mean()), "los_median": float(y.median()), "los_sd": float(y.std()),
        "los_min": float(y.min()), "los_max": float(y.max()), "los_skew": float(y.skew()),
        "los_histogram": {str(int(k)): int(v) for k, v in hist.items()},
        "group_means": {c: df.groupby(c)[TARGET].mean().round(3).to_dict() for c in SET_A_CATEGORICAL + ["Race"]},
        "group_counts": {c: df[c].value_counts().to_dict() for c in SET_A_CATEGORICAL + ["Race"]},
        "correlations_with_los": {c: float(np.corrcoef(df[c].astype(float), y)[0, 1])
                                  for c in SET_A_NUMERIC + SET_B_EXTRA_NUMERIC},
    }
