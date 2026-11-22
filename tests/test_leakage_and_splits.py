import numpy as np
import pytest

from backend.app.ml.features import FEATURE_SUBSETS_A, FORBIDDEN_PREDICTORS, FrequencyEncoder, feature_columns
from training.data import clean, load_raw, make_splits
from training.registry import MODELS, make


@pytest.mark.parametrize("subset", list(FEATURE_SUBSETS_A))
def test_set_a_excludes_post_discharge_and_identifiers(subset):
    cats, nums = feature_columns("A", subset)
    cols = set(cats + nums)
    assert not cols & FORBIDDEN_PREDICTORS
    assert not cols & {"Billing Amount", "Doctor", "Hospital", "Name", "Room Number", "Insurance Provider"}


def test_set_b_never_uses_target_or_discharge():
    cats, nums = feature_columns("B")
    assert not set(cats + nums) & FORBIDDEN_PREDICTORS


def test_frequency_encoder_is_fit_on_training_fold_only(signal_df):
    tr, va = signal_df.iloc[:1000], signal_df.iloc[1000:].copy()
    va["Doctor"] = "unseen-doctor"
    enc = FrequencyEncoder().fit(tr)
    assert (enc.transform(va)["doctor_freq"] == 0).all()


@pytest.mark.parametrize("key", ["gamma_glm", "xgboost", "catboost"])
def test_models_ignore_discharge_date(key, signal_df):
    """Scrambling Discharge Date must not change any prediction."""
    tr, te = signal_df.iloc[:1200], signal_df.iloc[1200:].copy()
    m = make(key, seed=1).fit(tr)
    p1 = m.predict(te)
    te["Discharge Date"] = te["Discharge Date"].sample(frac=1, random_state=0).values
    te["Billing Amount"] = 0.0
    np.testing.assert_allclose(p1, m.predict(te))


def test_real_data_holdout_is_disjoint_and_deduplicated():
    df, rep = clean(load_raw())
    assert rep["exact_duplicates_removed"] > 0
    key = df.assign(Name=df["Name"].str.lower()).drop(columns=["Length_of_Stay"])
    assert not key.duplicated().any()
    s = make_splits(df, persist=False)
    a = set(map(tuple, s.dev.assign(Name=s.dev.Name.str.lower()).astype(str).values))
    b = set(map(tuple, s.test.assign(Name=s.test.Name.str.lower()).astype(str).values))
    assert not a & b
    assert len(s.test) / len(df) == pytest.approx(0.2, abs=0.01)
    assert not set(s.dev_fit_idx) & set(s.dev_cal_idx)
    assert (df["Length_of_Stay"] >= 0).all()


def test_complexity_ordering_for_parsimony():
    order = sorted(MODELS, key=lambda k: MODELS[k].complexity)
    assert order[0] == "median_baseline" and order[-1] == "ft_transformer"
