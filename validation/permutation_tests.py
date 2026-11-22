"""Shuffled-target negative control and permutation test.

H0: the relationship between the features and LOS is no stronger than expected by chance.

Procedure (per model, with its final hyper-parameters):
1. Split the development data once into train / validation (fixed seed).
2. Fit on the real target, record validation MAE (``observed``).
3. Repeat n times: permute LOS across all development rows (breaking any feature-target
   link while preserving the marginal LOS distribution), refit, record validation MAE.
4. Empirical p-value = (#{shuffled MAE <= observed MAE} + 1) / (n + 1).

If real-target models are barely better than shuffled-target models, the dataset does not
contain usable predictive signal for LOS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from training.registry import make
from .metrics import mae


def permutation_test(model_key: str, dev: pd.DataFrame, params: dict, n_perm: int, seed: int,
                     model_kw: dict | None = None, log=print) -> dict:
    tr, va = train_test_split(np.arange(len(dev)), test_size=0.2, random_state=seed)
    real = make(model_key, seed=seed, **(model_kw or {})).fit(dev.iloc[tr], params)
    observed = mae(dev.iloc[va]["Length_of_Stay"], real.predict(dev.iloc[va]))
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_perm):
        shuffled = dev.copy()
        shuffled["Length_of_Stay"] = rng.permutation(shuffled["Length_of_Stay"].to_numpy())
        m = make(model_key, seed=seed + i + 1, **(model_kw or {})).fit(shuffled.iloc[tr], params)
        null.append(mae(shuffled.iloc[va]["Length_of_Stay"], m.predict(shuffled.iloc[va])))
    null = np.array(null)
    p = (np.sum(null <= observed) + 1) / (n_perm + 1) if n_perm else None
    res = {"model": model_key, "n_permutations": n_perm, "observed_mae": float(observed),
           "null_mae": null.tolist(),
           "null_mean": float(null.mean()) if n_perm else None,
           "null_sd": float(null.std(ddof=1)) if n_perm > 1 else None,
           "improvement_over_shuffled": float(null.mean() - observed) if n_perm else None,
           "p_value": float(p) if p is not None else None,
           "min_attainable_p": 1.0 / (n_perm + 1) if n_perm else None}
    if n_perm:
        log(f"  [perm] {model_key}: observed {observed:.3f} vs shuffled {null.mean():.3f}±{res['null_sd'] or 0:.3f} p={p:.3f}")
    return res
