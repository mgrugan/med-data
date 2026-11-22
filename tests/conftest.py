import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.ml.features import CATEGORY_LEVELS  # noqa: E402


def synthetic_frame(n=1500, seed=0, signal=True):
    """Frame with the derived (diabetes) schema. With signal=True, LOS depends on admission type."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in CATEGORY_LEVELS.items()})
    df["Age"] = rng.choice(np.arange(5, 100, 10), n).astype(float)
    for c in ["Prior Inpatient", "Prior Emergency", "Prior Outpatient", "Lab Procedures", "Procedures",
              "Medications Count", "Diagnoses Count"]:
        df[c] = rng.poisson(1.0, n).astype(float)
    df["Race"] = rng.choice(["Caucasian", "AfricanAmerican", "Other"], n)
    df["Discharge Disposition"] = rng.choice(["Home", "Rehabilitation"], n)
    df["Medication Change"] = rng.choice(["Changed", "No change"], n)
    df["Payer"] = rng.choice(["MC", "Unknown"], n)
    df["Medical Specialty"] = rng.choice(["Cardiology", "Unknown"], n)
    df["encounter_id"] = np.arange(n)
    df["patient_nbr"] = np.arange(n)
    base = df["Admission Type"].map({"Elective": 2, "Urgent": 5, "Emergency": 8, "Other/Unknown": 5}) if signal else 5
    df["Length_of_Stay"] = np.clip(np.round(base + rng.normal(0, 1.5, n)), 1, 14).astype(float)
    return df


@pytest.fixture
def signal_df():
    return synthetic_frame(signal=True)


@pytest.fixture
def null_df():
    return synthetic_frame(signal=False)
