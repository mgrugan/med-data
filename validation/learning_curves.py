"""Learning curves: training vs validation MAE at 10/25/50/75/100% of the training data.

Interpretation guide
* Both errors high and converged       -> under-fitting / no learnable signal.
* Large train-validation gap           -> over-fitting.
* Validation error still falling at 100% -> more data would likely help.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from training.registry import make
from .metrics import mae

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def learning_curve(model_key: str, dev: pd.DataFrame, params: dict, seeds: int, seed: int,
                   model_kw: dict | None = None) -> dict:
    tr, va = train_test_split(np.arange(len(dev)), test_size=0.2, random_state=seed)
    d_va = dev.iloc[va]
    points = []
    for f in FRACTIONS:
        tr_err, va_err = [], []
        for s in range(seeds):
            rng = np.random.default_rng(seed + s)
            sub = rng.choice(tr, size=max(50, int(round(f * len(tr)))), replace=False)
            d_tr = dev.iloc[np.sort(sub)]
            m = make(model_key, seed=seed + s, **(model_kw or {})).fit(d_tr, params)
            tr_err.append(mae(d_tr["Length_of_Stay"], m.predict(d_tr)))
            va_err.append(mae(d_va["Length_of_Stay"], m.predict(d_va)))
        points.append({"fraction": f, "n_train": int(round(f * len(tr))),
                       "train_mae": float(np.mean(tr_err)), "train_sd": float(np.std(tr_err)),
                       "val_mae": float(np.mean(va_err)), "val_sd": float(np.std(va_err))})
    last = points[-1]
    gap = last["val_mae"] - last["train_mae"]
    slope = points[-2]["val_mae"] - last["val_mae"]
    if gap > 0.5:
        diag = "overfitting: training error is well below validation error"
    elif slope > 0.05:
        diag = "validation error still decreasing; more data may help"
    else:
        diag = "converged with small train/validation gap; extra data unlikely to help (little learnable signal)"
    return {"model": model_key, "points": points, "final_gap": gap, "diagnosis": diag}
