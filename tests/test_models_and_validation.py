"""The framework must detect signal when it exists and report none when it does not."""
import numpy as np

from training.registry import make
from validation.metrics import mae
from validation.nested_cv import nested_cv
from validation.permutation_tests import permutation_test
from validation.tuning import tune


def test_permutation_test_detects_real_signal(signal_df):
    r = permutation_test("gamma_glm", signal_df, {}, n_perm=9, seed=0, log=lambda *_: None)
    assert r["p_value"] == 0.1  # (0 + 1) / (9 + 1): every shuffled model is worse
    assert r["improvement_over_shuffled"] > 2


def test_permutation_test_on_null_data_is_not_significant(null_df):
    r = permutation_test("gamma_glm", null_df, {}, n_perm=19, seed=0, log=lambda *_: None)
    assert r["p_value"] > 0.05


def test_permutation_p_value_formula(null_df):
    r = permutation_test("median_baseline", null_df, {}, n_perm=4, seed=0, log=lambda *_: None)
    k = sum(v <= r["observed_mae"] for v in r["null_mae"])
    assert r["p_value"] == (k + 1) / 5


def test_nested_cv_structure_and_no_outer_leakage(signal_df):
    res = nested_cv("xgboost", signal_df, outer_folds=3, inner_folds=2, n_trials=2, seed=0, log=lambda *_: None)
    assert len(res["folds"]) == 3 and len(res["studies"]) == 3
    assert all(s["n_folds"] == 2 for s in res["studies"])
    assert res["inner_fits"] == 3 * 2 * 2
    # inner searches only ever saw the outer-training rows
    assert all(f["n_train"] + f["n_val"] == len(signal_df) for f in res["folds"])


def test_tuning_stores_trials_and_seed(signal_df):
    st = tune("catboost", signal_df.iloc[:600], n_folds=2, n_trials=2, seed=3)
    assert st["seed"] == 3 and len(st["trials"]) == 2 and "best_params" in st
    assert all("inner_cv_mae" in t for t in st["trials"])


def test_models_beat_median_when_signal_exists(signal_df):
    tr, te = signal_df.iloc[:1200], signal_df.iloc[1200:]
    y = te["Length_of_Stay"]
    base = mae(y, make("median_baseline").fit(tr).predict(te))
    for k in ["gamma_glm", "xgboost", "catboost"]:
        assert mae(y, make(k, seed=0).fit(tr).predict(te)) < base - 1.5, k


def test_glm_reports_likelihood_quantities(signal_df):
    m = make("gamma_glm").fit(signal_df)
    ic = m.information_criteria()
    assert ic["aicc"] > ic["aic"] and ic["bic"] > ic["aic"]
    assert m.null_comparison()["p_value"] < 1e-6
    assert all({"coef", "std_err", "ci_low", "ci_high"} <= set(r) for r in m.coefficient_table())
