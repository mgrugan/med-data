"""Survival-style metrics for distributional LOS models (all stays are observed events).

* Harrell's C-index: P(pred_i < pred_j | T_i < T_j), prediction ties count 1/2.
* Brier score over time and the Integrated Brier Score (IBS) on a day grid, with the
  Kaplan-Meier (feature-free) curve as reference. With no censoring, KM equals the
  empirical survival function, and inverse-probability-of-censoring weights are all 1.
* Calibration over time: predicted vs observed P(still hospitalised at day t).
* Negative log-likelihood (mean -log density at the observed LOS).
"""
from __future__ import annotations

import numpy as np


def concordance_index(t: np.ndarray, pred: np.ndarray, chunk: int = 512) -> float:
    t, pred = np.asarray(t, float), np.asarray(pred, float)
    conc = disc = ties = 0.0
    for i in range(0, len(t), chunk):
        ti, pi = t[i:i + chunk, None], pred[i:i + chunk, None]
        comparable = ti < t[None, :]
        dp = pi - pred[None, :]
        conc += np.sum(comparable & (dp < 0))
        disc += np.sum(comparable & (dp > 0))
        ties += np.sum(comparable & (dp == 0))
    total = conc + disc + ties
    return float((conc + 0.5 * ties) / total) if total else float("nan")


def empirical_survival(t_train: np.ndarray, grid: np.ndarray) -> np.ndarray:
    t_train = np.sort(np.asarray(t_train, float))
    return 1.0 - np.searchsorted(t_train, grid, side="right") / len(t_train)


def brier_curve(t: np.ndarray, surv: np.ndarray, grid: np.ndarray) -> np.ndarray:
    alive = (np.asarray(t)[:, None] > grid[None, :]).astype(float)
    return np.mean((alive - surv) ** 2, axis=0)


def integrated_brier(t, surv, grid) -> float:
    b = brier_curve(t, surv, grid)
    return float(np.trapezoid(b, grid) / (grid[-1] - grid[0]))


def calibration_over_time(t, surv, grid, days=(5, 10, 15, 20, 25), bins=5) -> list[dict]:
    t = np.asarray(t)
    out = []
    for d in days:
        j = int(np.argmin(np.abs(grid - d)))
        s = surv[:, j]
        edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
        rows = []
        for b in range(len(edges) - 1):
            m = which == b
            if m.any():
                rows.append({"predicted": float(s[m].mean()), "observed": float(np.mean(t[m] > d)), "n": int(m.sum())})
        out.append({"day": int(d), "bins": rows,
                    "mean_predicted": float(s.mean()), "observed": float(np.mean(t > d))})
    return out


def survival_report(t_test, pred_point, surv, log_density, t_train, grid) -> dict:
    km = np.broadcast_to(empirical_survival(t_train, grid), surv.shape)
    return {
        "c_index": concordance_index(t_test, pred_point),
        "integrated_brier_score": integrated_brier(t_test, surv, grid),
        "ibs_reference_kaplan_meier": integrated_brier(t_test, km, grid),
        "brier_curve": {"days": grid.tolist(), "model": brier_curve(t_test, surv, grid).tolist(),
                        "kaplan_meier": brier_curve(t_test, km, grid).tolist()},
        "negative_log_likelihood": float(-np.mean(log_density)),
        "calibration_over_time": calibration_over_time(t_test, surv, grid),
    }
