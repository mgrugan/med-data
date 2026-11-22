import numpy as np
import pytest

from validation import bootstrap_validation as bv
from validation.calibration import calibration_report
from validation.conformal_prediction import conformal_quantile, evaluate
from validation.metrics import all_metrics
from validation.model_comparison import corrected_resampled_ttest
from validation.survival_metrics import concordance_index, empirical_survival, integrated_brier


def test_metrics_basic():
    m = all_metrics([1, 2, 3], [1, 2, 4])
    assert m["mae"] == pytest.approx(1 / 3) and m["bias"] == pytest.approx(1 / 3) and m["medae"] == 0


def test_bootstrap_ci_contains_point_and_is_reasonable():
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 30, 3000)
    p = y + rng.normal(0, 3, 3000)
    out = bv.bootstrap_model(y, p, B=400, seed=1)["metrics"]["mae"]
    lo, hi = out["ci_percentile"]
    blo, bhi = out["ci_bca"]
    assert lo < out["estimate"] < hi and blo < out["estimate"] < bhi
    assert hi - lo < 0.3


def test_paired_difference_of_identical_models_is_zero():
    rng = np.random.default_rng(0)
    y, p = rng.uniform(1, 30, 500), rng.uniform(1, 30, 500)
    d = bv.bootstrap_difference(y, p, p, B=200, seed=0)["mae"]
    assert d["difference"] == 0 and d["ci_percentile"] == [0, 0] and not d["excludes_zero"]


def test_paired_difference_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 30, 2000)
    good, bad = y + rng.normal(0, 2, 2000), y + rng.normal(0, 6, 2000)
    d = bv.bootstrap_difference(y, good, bad, B=300, seed=0)["mae"]
    assert d["ci_percentile"][1] < 0 and d["excludes_zero"]


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9, 0.95])
def test_split_conformal_coverage(level):
    rng = np.random.default_rng(1)
    y_cal, y_te = rng.uniform(1, 30, 4000), rng.uniform(1, 30, 20000)
    p_cal, p_te = np.full(4000, 15.5), np.full(20000, 15.5)
    res = evaluate(y_cal, p_cal, y_te, p_te, [level], y_dev_marginal=y_cal)["levels"][str(level)]
    assert res["coverage"] == pytest.approx(level, abs=0.02)


def test_conformal_quantile_finite_sample_rank():
    assert conformal_quantile(np.arange(1, 10), 0.9) == 9  # ceil(10*0.9)=9th smallest


def test_corrected_ttest_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.05, 0.05, 15)
    r = corrected_resampled_ttest(d, 35000, 9000)
    assert r["p_value"] > r["naive_p_value"]


def test_concordance_index_bounds():
    t = np.arange(1, 101, dtype=float)
    assert concordance_index(t, t) == 1.0
    assert concordance_index(t, -t) == 0.0
    assert concordance_index(t, np.zeros_like(t)) == 0.5


def test_ibs_of_true_marginal_is_lower_than_bad_curve():
    rng = np.random.default_rng(0)
    t = rng.integers(1, 31, 3000).astype(float)
    grid = np.arange(1.0, 30.0)
    km = np.broadcast_to(empirical_survival(t, grid), (len(t), len(grid)))
    bad = np.full_like(km, 0.99)
    assert integrated_brier(t, km, grid) < integrated_brier(t, bad, grid)


def test_calibration_slope_one_for_perfect_predictions():
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 30, 2000)
    assert calibration_report(y, y)["calibration_slope"] == pytest.approx(1.0)
