import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.ml.features import CATEGORY_LEVELS  # noqa: E402


def synthetic_frame(n=1500, seed=0, signal=True):
    """Frame with the dataset's schema. With signal=True, LOS depends on admission type."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in CATEGORY_LEVELS.items()})
    df["Age"] = rng.integers(18, 90, n).astype(float)
    df["Name"] = [f"p{i}" for i in range(n)]
    df["Doctor"] = [f"d{i % 300}" for i in range(n)]
    df["Hospital"] = [f"h{i % 200}" for i in range(n)]
    df["Room Number"] = rng.integers(100, 500, n)
    df["Billing Amount"] = rng.uniform(1000, 50000, n)
    df["Date of Admission"] = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    base = df["Admission Type"].map({"Elective": 6, "Urgent": 12, "Emergency": 20}) if signal else 13
    df["Length_of_Stay"] = np.clip(np.round(base + rng.normal(0, 3, n)), 1, 30).astype(float)
    df["Discharge Date"] = df["Date of Admission"] + pd.to_timedelta(df["Length_of_Stay"], unit="D")
    return df


@pytest.fixture
def signal_df():
    return synthetic_frame(signal=True)


@pytest.fixture
def null_df():
    return synthetic_frame(signal=False)
