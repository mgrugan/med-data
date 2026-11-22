"""Calibration of point predictions: predicted vs observed LOS within bins."""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm

FIXED_BINS = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 10), (10, 15)]


def calibration_report(y, p, n_bins: int = 10) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    quantile_bins = []
    for b in range(max(len(edges) - 1, 1)):
        m = which == b
        if m.any():
            quantile_bins.append({"pred_low": float(p[m].min()), "pred_high": float(p[m].max()),
                                  "mean_predicted": float(p[m].mean()), "mean_observed": float(y[m].mean()),
                                  "se_observed": float(y[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else None,
                                  "n": int(m.sum())})
    fixed = []
    for lo, hi in FIXED_BINS:
        m = (p >= lo) & (p < hi)
        if m.any():
            fixed.append({"range": f"{lo}-{hi} days", "mean_predicted": float(p[m].mean()),
                          "mean_observed": float(y[m].mean()), "n": int(m.sum())})
    # Calibration slope/intercept: regress observed on predicted. Ideal = slope 1, intercept 0.
    if np.std(p) > 1e-9:
        ols = sm.OLS(y, sm.add_constant(p)).fit()
        slope, se = float(ols.params[1]), float(ols.bse[1])
        intercept = float(ols.params[0])
        slope_ci = [slope - 1.96 * se, slope + 1.96 * se]
    else:
        slope = se = intercept = float("nan")
        slope_ci = [float("nan"), float("nan")]
    return {"quantile_bins": quantile_bins, "fixed_bins": fixed, "calibration_slope": slope,
            "calibration_slope_se": se, "calibration_slope_ci": slope_ci, "calibration_intercept": intercept,
            "calibration_in_the_large": float(np.mean(y) - np.mean(p)),
            "prediction_sd": float(np.std(p)), "observed_sd": float(np.std(y)),
            "prediction_range": [float(p.min()), float(p.max())]}
