"""Subgroup error analysis on the final test set.

Differences between subgroups describe where the *model* is systematically weaker. The
dataset is synthetic, so they must not be read as biological or clinical truths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.features import age_group

GROUP_COLUMNS = ["Gender", "Age group", "Medical Condition", "Admission Type", "Medication", "Test Results"]


def subgroup_errors(test: pd.DataFrame, pred: np.ndarray, B: int = 200, seed: int = 0) -> dict:
    df = test.assign(**{"Age group": age_group(test["Age"]).values, "_pred": pred})
    df["_abs"] = np.abs(df["_pred"] - df["Length_of_Stay"])
    df["_err"] = df["_pred"] - df["Length_of_Stay"]
    overall = float(df["_abs"].mean())
    rng = np.random.default_rng(seed)
    out = {"overall_mae": overall, "groups": {}}
    for col in GROUP_COLUMNS:
        rows = []
        for lvl, g in df.groupby(col):
            a = g["_abs"].to_numpy()
            boots = a[rng.integers(0, len(a), size=(B, len(a)))].mean(axis=1)
            rows.append({"level": str(lvl), "n": int(len(g)), "mae": float(a.mean()),
                         "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "bias": float(g["_err"].mean()), "mean_observed": float(g["Length_of_Stay"].mean()),
                         "mean_predicted": float(g["_pred"].mean()), "delta_vs_overall": float(a.mean() - overall)})
        out["groups"][col] = rows
    return out


def residuals_by_observed(y, p) -> list[dict]:
    """Residual pattern vs observed LOS: a no-signal model over-predicts short stays and
    under-predicts long stays (regression to the mean)."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for d in np.unique(y):
        m = y == d
        rows.append({"observed": float(d), "mean_residual": float(np.mean(p[m] - y[m])),
                     "mean_predicted": float(p[m].mean()), "n": int(m.sum())})
    return rows
