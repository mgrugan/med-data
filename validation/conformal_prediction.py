"""Split conformal prediction intervals.

Final models are fitted on ``dev_fit`` (85% of development data). Absolute residuals on the
disjoint ``dev_cal`` set (15%), never used for fitting or early stopping, give the
conformal quantile

    q = ceil((n_cal + 1)(1 - alpha)) / n_cal empirical quantile of |y - ŷ|

and the interval  [max(0, ŷ - q), ŷ + q]  has finite-sample marginal coverage >= 1 - alpha
under exchangeability. Coverage and width are then evaluated on the untouched test set,
overall and within subgroups (conditional coverage is NOT guaranteed).

A "no-information" reference interval (marginal quantiles of LOS, ignoring all features)
shows whether the features make intervals any narrower.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.features import age_group


def conformal_quantile(residuals: np.ndarray, level: float) -> float:
    r = np.sort(np.abs(np.asarray(residuals, float)))
    n = len(r)
    k = int(np.ceil((n + 1) * level))
    return float(r[min(k, n) - 1])


def interval(pred: np.ndarray, q: float, method: str = "absolute") -> tuple[np.ndarray, np.ndarray]:
    """absolute: symmetric  pred ± q.
    log: multiplicative  expm1(log1p(pred) ± q)  - asymmetric, wider above than below, which
         suits a right-skewed target such as LOS. Coverage guarantee is unchanged because the
         conformity score |log1p(y) - log1p(pred)| is just a different (monotone) score."""
    pred = np.asarray(pred, float)
    eps = 1e-9  # LOS is integer-valued: many outcomes sit exactly on an endpoint; avoid round-off exclusion
    if method == "log":
        z = np.log1p(np.maximum(pred, 0))
        return np.maximum(0.0, np.expm1(z - q) - eps), np.expm1(z + q) + eps
    return np.maximum(0.0, pred - q - eps), pred + q + eps


def scores(y, p, method: str = "absolute") -> np.ndarray:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if method == "log":
        return np.abs(np.log1p(y) - np.log1p(np.maximum(p, 0)))
    return np.abs(y - p)


def evaluate(y_cal, p_cal, y_test, p_test, levels, test_df: pd.DataFrame | None = None,
             y_dev_marginal=None, method: str = "absolute") -> dict:
    y_test, p_test = np.asarray(y_test, float), np.asarray(p_test, float)
    out = {"n_calibration": int(len(y_cal)), "method": method, "levels": {}}
    for lv in levels:
        q = conformal_quantile(scores(y_cal, p_cal, method), lv)
        lo, hi = interval(p_test, q, method)
        cover = (y_test >= lo) & (y_test <= hi)
        res = {"q": q, "coverage": float(cover.mean()), "mean_width": float(np.mean(hi - lo)),
               "median_width": float(np.median(hi - lo)), "nominal": lv}
        if y_dev_marginal is not None:
            a = (1 - lv) / 2
            mlo, mhi = np.quantile(y_dev_marginal, [a, 1 - a])
            res["no_information_interval"] = [float(mlo), float(mhi)]
            res["no_information_width"] = float(mhi - mlo)
            res["no_information_coverage"] = float(np.mean((y_test >= mlo) & (y_test <= mhi)))
            res["width_ratio_vs_no_information"] = float(res["mean_width"] / max(mhi - mlo, 1e-9))
        if test_df is not None:
            groups = {}
            frame = test_df.assign(_cover=cover, _age=age_group(test_df["Age"]).values)
            for col in ["Medical Condition", "Admission Type", "Gender", "Race", "_age"]:
                g = frame.groupby(col)["_cover"].agg(["mean", "size"])
                groups["Age group" if col == "_age" else col] = {
                    str(k): {"coverage": float(v["mean"]), "n": int(v["size"])} for k, v in g.iterrows()}
            res["conditional_coverage"] = groups
        out["levels"][str(lv)] = res
    return out
