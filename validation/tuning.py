"""Optuna hyper-parameter search evaluated by K-fold CV *within the data it is given*.

When called from nested CV, ``df`` is the outer-training portion only, so the outer
validation fold is never seen by the search. The final test set is never passed here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from training.registry import make
from .metrics import mae

optuna.logging.set_verbosity(optuna.logging.WARNING)


def cv_score(model_key: str, df: pd.DataFrame, params: dict, n_folds: int, seed: int,
             model_kw: dict | None = None) -> tuple[float, list[float]]:
    scores = []
    for k, (tr, va) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(df)):
        m = make(model_key, seed=seed + k, **(model_kw or {})).fit(df.iloc[tr], params)
        scores.append(mae(df.iloc[va]["Length_of_Stay"], m.predict(df.iloc[va])))
    return float(np.mean(scores)), scores


def tune(model_key: str, df: pd.DataFrame, n_folds: int, n_trials: int, seed: int,
         model_kw: dict | None = None, context: str = "") -> dict[str, Any]:
    proto = make(model_key, seed=seed, **(model_kw or {}))
    grid = proto.grid()
    if grid is not None:
        sampler = optuna.samplers.GridSampler(grid, seed=seed)
        n_trials = int(np.prod([len(v) for v in grid.values()]))
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(5, n_trials))
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"{model_key}{context}")
    # Trial 0 = sensible defaults, so the search can never do worse than the defaults.
    if grid is None:
        study.enqueue_trial(proto.default_params())

    def objective(trial):
        params = proto.suggest(trial)
        score, folds = cv_score(model_key, df, params, n_folds, seed, model_kw)
        trial.set_user_attr("fold_mae", folds)
        return score

    study.optimize(objective, n_trials=n_trials)
    best = {**proto.default_params(), **study.best_params}
    if model_key == "ft_transformer":
        best["n_heads"] = 4
    return {
        "model": model_key, "context": context, "seed": seed, "n_folds": n_folds, "n_trials": len(study.trials),
        "sampler": type(sampler).__name__, "best_params": best, "best_inner_cv_mae": float(study.best_value),
        "trials": [{"number": t.number, "params": t.params, "inner_cv_mae": t.value,
                    "fold_mae": t.user_attrs.get("fold_mae")} for t in study.trials],
    }
