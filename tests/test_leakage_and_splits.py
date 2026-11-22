import numpy as np
import pandas as pd
import pytest

from backend.app.ml.features import (FEATURE_SUBSETS_A, FORBIDDEN_PREDICTORS, LEAKAGE_RISK_FEATURES,
                                     feature_columns)
from training.data import DEATH_OR_HOSPICE, clean, icd9_group, load_raw, make_splits
from training.registry import MODELS, make


@pytest.mark.parametrize("subset", list(FEATURE_SUBSETS_A))
def test_set_a_excludes_leakage_and_identifiers(subset):
    cats, nums = feature_columns("A", subset)
    cols = set(cats + nums)
    assert not cols & FORBIDDEN_PREDICTORS
    assert not cols & LEAKAGE_RISK_FEATURES
    assert "Race" not in cols


def test_set_b_never_uses_target_or_identifiers():
    cats, nums = feature_columns("B")
    assert not set(cats + nums) & FORBIDDEN_PREDICTORS
    assert LEAKAGE_RISK_FEATURES <= set(cats + nums)


@pytest.mark.parametrize("key", ["gamma_glm", "xgboost", "catboost"])
def test_set_a_models_ignore_during_stay_and_discharge_variables(key, signal_df):
    """Scrambling every leakage-risk column must not change any set-A prediction."""
    tr, te = signal_df.iloc[:1200], signal_df.iloc[1200:].copy()
    m = make(key, seed=1).fit(tr)
    p1 = m.predict(te)
    for c in LEAKAGE_RISK_FEATURES | {"Race"}:
        te[c] = te[c].sample(frac=1, random_state=0).values
    np.testing.assert_allclose(p1, m.predict(te))


def test_icd9_grouping():
    assert icd9_group("250.83") == "Diabetes" and icd9_group("428") == "Circulatory"
    assert icd9_group("486") == "Respiratory" and icd9_group("820") == "Injury"
    assert icd9_group("V57") == "Other" and icd9_group(np.nan) == "Other" and icd9_group("174") == "Neoplasms"


def test_cohort_one_encounter_per_patient_no_deaths_and_disjoint_holdout():
    raw = load_raw()
    df, rep = clean(raw)
    assert df["patient_nbr"].is_unique
    assert rep["repeat_encounters_removed"] > 0 and rep["death_or_hospice_removed"] > 0
    # the kept encounter is each patient's earliest
    first = raw.sort_values("encounter_id").drop_duplicates("patient_nbr")
    kept = df.set_index("patient_nbr")["encounter_id"]
    assert (kept == first.set_index("patient_nbr").loc[kept.index, "encounter_id"]).all()
    s = make_splits(df, persist=False)
    assert not set(s.dev["patient_nbr"]) & set(s.test["patient_nbr"])
    assert len(s.test) / len(df) == pytest.approx(0.2, abs=0.01)
    assert not set(s.dev_fit_idx) & set(s.dev_cal_idx)
    assert df["Length_of_Stay"].between(1, 14).all()


def test_complexity_ordering_for_parsimony():
    order = sorted(MODELS, key=lambda k: MODELS[k].complexity)
    assert order[0] == "median_baseline" and order[-1] == "ft_transformer"
