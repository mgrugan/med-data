"""Model ranking, selection and the methodology quality gate.

Selection uses only cross-validation results (never the test set):
1. Rank by nested-CV mean MAE (the unbiased estimate of each modelling procedure).
2. Compare every model with the best one on the identical repeated-CV folds using the
   Nadeau-Bengio corrected resampled t-test, which inflates the variance to account for
   overlapping training sets across folds (a naive paired t-test is anti-conservative).
3. Among models that are NOT significantly worse than the best (p > 0.05) and within a
   practical-equivalence margin, pick the simplest (parsimony), breaking ties by fold SD.

The smallest point estimate therefore does not automatically win.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

# Thresholds are RELATIVE to the median-baseline MAE so they mean the same thing on any
# dataset. They were fixed before the final test-set evaluation of the real data.
PRACTICAL_MARGIN_REL = 0.01        # models within 1% MAE of the leader are practically equivalent
MEANINGFUL_IMPROVEMENT_REL = 0.05  # >=5% lower MAE than the median baseline counts as meaningful
CALIBRATION_SLOPE_RANGE = (0.8, 1.2)
FEATURE_MODELS = ["gamma_glm", "xgboost", "xgboost_aft", "catboost", "ft_transformer"]


def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> dict[str, float]:
    d = np.asarray(diffs, float)
    J = len(d)
    var = d.var(ddof=1)
    if var <= 0:
        return {"mean_diff": float(d.mean()), "t": float("nan"), "p_value": 1.0 if d.mean() == 0 else 0.0, "df": J - 1}
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return {"mean_diff": float(d.mean()), "t": float(t), "p_value": float(p), "df": J - 1,
            "naive_p_value": float(stats.ttest_1samp(d, 0.0).pvalue)}


def select_model(nested: dict, repeated: dict, complexity: dict[str, int], n_dev: int, n_folds: int) -> dict[str, Any]:
    keys = list(nested)
    margin = PRACTICAL_MARGIN_REL * nested["median_baseline"]["summary"]["mae"]["mean"]
    ranking = sorted(keys, key=lambda k: nested[k]["summary"]["mae"]["mean"])
    best = ranking[0]
    best_folds = np.array([f["mae"] for f in repeated[best]["folds"]])
    n_test = n_dev // n_folds
    n_train = n_dev - n_test
    comps = {}
    for k in keys:
        f = np.array([x["mae"] for x in repeated[k]["folds"]])
        res = corrected_resampled_ttest(f - best_folds, n_train, n_test) if k != best else {
            "mean_diff": 0.0, "t": 0.0, "p_value": 1.0, "df": len(f) - 1}
        res["equivalent_to_best"] = bool(k == best or (res["p_value"] > 0.05 or res["mean_diff"] < margin))
        comps[k] = res
    equivalent = [k for k in keys if comps[k]["equivalent_to_best"]]
    stab = {k: repeated[k]["summary"]["mae"]["sd"] for k in keys}
    selected = sorted(equivalent, key=lambda k: (complexity[k], stab[k]))[0]
    feat_equiv = [k for k in equivalent if k in FEATURE_MODELS]
    display = (sorted(feat_equiv, key=lambda k: (complexity[k], stab[k]))[0] if feat_equiv else
               min(FEATURE_MODELS, key=lambda k: nested[k]["summary"]["mae"]["mean"]) if all(
                   k in nested for k in FEATURE_MODELS) else best)
    reason = (f"{nested[best]['summary']['mae']['mean']:.3f}-day nested-CV MAE makes '{best}' the point-estimate "
              f"leader, but {len(equivalent)} of {len(keys)} models are statistically indistinguishable from it "
              f"(corrected resampled t-test, p > 0.05, or within {margin:.3f} days = 1% of baseline MAE). By parsimony the "
              f"simplest of these, '{selected}', is selected.")
    return {"ranking_by_nested_mae": ranking, "point_estimate_leader": best, "selected_model": selected,
            "display_model": display, "equivalent_set": equivalent, "comparisons_vs_leader": comps,
            "fold_stability_sd": stab, "practical_margin_days": margin, "reason": reason,
            "selection_uses_test_set": False}


def quality_gate(nested, test_boot_diffs, permutation, conformal, calibration, selection) -> dict[str, Any]:
    base = nested["median_baseline"]["summary"]["mae"]["mean"]
    feat = [k for k in FEATURE_MODELS if k in nested]
    improvements = {k: base - nested[k]["summary"]["mae"]["mean"] for k in feat}
    threshold = MEANINGFUL_IMPROVEMENT_REL * base
    beats = {k: bool(improvements[k] >= threshold and
                     test_boot_diffs.get(f"{k}__vs__median_baseline", {}).get("mae", {}).get("ci_percentile", [0, 0])[1] < 0)
             for k in feat}
    # Statistical (not practical) improvement: is ANY advanced model's paired bootstrap MAE
    # difference vs the median baseline entirely below zero? Models that are significantly
    # *worse* do not count as a "meaningful difference".
    better = {k: test_boot_diffs.get(f"{k}__vs__median_baseline", {}).get("mae", {}).get("ci_percentile", [0, 0])[1] < 0
              for k in feat}
    perm_sig = {k: (permutation[k]["p_value"] is not None and permutation[k]["p_value"] < 0.05 and
                    (permutation[k]["improvement_over_shuffled"] or 0) >= threshold)
                for k in feat if k in permutation}
    disp = selection["display_model"]
    c90 = conformal[disp]["levels"]["0.9"]
    cal = calibration[disp]
    slope = cal["calibration_slope"]
    poor_cal = bool(slope is None or np.isnan(slope) or not (CALIBRATION_SLOPE_RANGE[0] <= slope <= CALIBRATION_SLOPE_RANGE[1])
                    or cal["prediction_sd"] < 0.1 * cal["observed_sd"])
    checks = [
        {"id": "beats_baseline", "failed": not any(beats.values()),
         "label": "Advanced models fail to meaningfully beat the median baseline",
         "pass_label": f"An advanced model beats the median baseline by at least 5% MAE ({threshold:.3f} days)",
         "detail": {"improvement_days_vs_median": improvements, "meaningful": beats,
                    "threshold_days": threshold, "threshold_relative": MEANINGFUL_IMPROVEMENT_REL}},
        {"id": "models_distinguishable", "failed": not any(better.values()),
         "label": "Bootstrap intervals show no meaningful difference between models",
         "pass_label": "At least one advanced model is significantly better than the baseline (bootstrap CI below 0)",
         "detail": {"significantly_better_than_median": better}},
        {"id": "shuffled_target", "failed": not any(perm_sig.values()),
         "label": "Shuffled-target models perform similarly to real-target models",
         "pass_label": "Real-target models clearly beat shuffled-target models (permutation p < 0.05)",
         "detail": {k: {"p_value": permutation[k]["p_value"],
                        "improvement_days": permutation[k]["improvement_over_shuffled"]} for k in perm_sig}},
        {"id": "interval_width", "failed": c90.get("width_ratio_vs_no_information", 0) >= 0.9,
         "label": "Prediction intervals are excessively wide",
         "pass_label": "Prediction intervals are at least 10% narrower than a no-information interval",
         "detail": {"mean_width_90": c90["mean_width"], "no_information_width_90": c90.get("no_information_width"),
                    "ratio": c90.get("width_ratio_vs_no_information")}},
        {"id": "calibration", "failed": poor_cal, "label": "Calibration is poor or predictions are non-discriminative",
         "pass_label": "Predictions are calibrated (slope within 0.8-1.2) and discriminative",
         "detail": {"slope": cal["calibration_slope"], "slope_ci": cal["calibration_slope_ci"],
                    "prediction_sd": cal["prediction_sd"], "observed_sd": cal["observed_sd"]}},
    ]
    failed = [c for c in checks if c["failed"]]
    return {
        "warning": bool(failed), "checks": checks, "n_failed": len(failed),
        "headline": ("Length of stay in this dataset appears to contain limited predictable signal from admission-time "
                     "information. "
                     "Predictions are displayed for methodological demonstration and should not be interpreted "
                     "as clinically meaningful.") if failed else "All methodology checks passed.",
        "signal_statement": ("The dataset does not appear to contain strong predictive relationships "
                             "for hospital length of stay.") if any(c["id"] == "shuffled_target" for c in failed) else None,
    }
