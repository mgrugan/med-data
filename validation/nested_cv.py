"""Nested K-fold cross-validation.

Outer loop (K_outer folds) estimates generalisation error of the *whole modelling
procedure* (preprocessing + feature-subset choice + hyper-parameter search + fit).
Inner loop (K_inner folds, run separately inside every outer-training portion) performs
the Optuna search, feature-subset selection and preprocessing decisions.

The outer validation fold is never used for tuning, early stopping or preprocessing.
All models share the same outer folds, so outer-fold scores are paired across models.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from training.registry import TUNABLE, make
from .metrics import all_metrics
from .tuning import tune


def nested_cv(model_key: str, dev: pd.DataFrame, outer_folds: int, inner_folds: int, n_trials: int,
              seed: int, model_kw: dict | None = None, log=print) -> dict[str, Any]:
    outer = KFold(outer_folds, shuffle=True, random_state=seed)
    folds, studies = [], []
    oof = np.full(len(dev), np.nan)
    for k, (tr, va) in enumerate(outer.split(dev)):
        t0 = time.time()
        d_tr, d_va = dev.iloc[tr], dev.iloc[va]
        if model_key in TUNABLE:
            st = tune(model_key, d_tr, inner_folds, n_trials, seed + 100 * (k + 1), model_kw, context=f"_outer{k}")
            params = st["best_params"]
            studies.append(st)
        else:
            params, st = {}, None
        m = make(model_key, seed=seed + k, **(model_kw or {})).fit(d_tr, params)
        pred = m.predict(d_va)
        oof[va] = pred
        met = all_metrics(d_va["Length_of_Stay"], pred)
        folds.append({"outer_fold": k, "n_train": len(tr), "n_val": len(va), "params": params,
                      "inner_best_mae": st["best_inner_cv_mae"] if st else None,
                      "best_iteration": m.best_iteration(), "seconds": time.time() - t0, **met})
        log(f"  [nested] {model_key} outer {k}: MAE={met['mae']:.3f} ({time.time() - t0:.0f}s)")
    return {"model": model_key, "outer_folds": outer_folds, "inner_folds": inner_folds,
            "trials_per_inner_search": n_trials,
            "inner_fits": outer_folds * inner_folds * n_trials if model_key in TUNABLE else 0,
            "folds": folds, "summary": summarize([f for f in folds]), "studies": studies,
            "oof_pred": oof.tolist()}


def summarize(folds: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for metric in ["mae", "rmse", "medae", "r2", "mape", "bias"]:
        v = np.array([f[metric] for f in folds], float)
        q1, q3 = np.percentile(v, [25, 75])
        out[metric] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                       "median": float(np.median(v)), "q1": float(q1), "q3": float(q3),
                       "iqr": float(q3 - q1), "min": float(v.min()), "max": float(v.max()),
                       "values": v.tolist()}
    return out
