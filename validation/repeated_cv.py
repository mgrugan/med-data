"""Repeated K-fold CV of the *final* configuration of each model.

A single 5-fold partition gives an unstable estimate, so the split is repeated with
different seeds. Every model is evaluated on the identical (repeat, fold) partitions,
which makes fold-level differences paired and enables the corrected resampled t-test in
``model_comparison``.

Note: the final hyper-parameters were selected by CV on the same development data, so
these numbers are slightly optimistic relative to the nested-CV estimate. The nested CV
result remains the headline generalisation estimate.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from training.registry import make
from .metrics import all_metrics
from .nested_cv import summarize


def repeated_cv(model_key: str, dev: pd.DataFrame, params: dict, n_folds: int, repeats: int, seed: int,
                model_kw: dict | None = None, log=print) -> dict[str, Any]:
    folds = []
    for r in range(repeats):
        for k, (tr, va) in enumerate(KFold(n_folds, shuffle=True, random_state=seed + 1000 * r).split(dev)):
            m = make(model_key, seed=seed + 10 * r + k, **(model_kw or {})).fit(dev.iloc[tr], params)
            met = all_metrics(dev.iloc[va]["Length_of_Stay"], m.predict(dev.iloc[va]))
            folds.append({"repeat": r, "fold": k, "train_mae": None, **met})
        log(f"  [repeated] {model_key} repeat {r}: MAE={np.mean([f['mae'] for f in folds[-n_folds:]]):.3f}")
    return {"model": model_key, "n_folds": n_folds, "repeats": repeats, "params": params,
            "folds": folds, "summary": summarize(folds)}
