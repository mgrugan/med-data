"""Explainability: TreeSHAP for the boosted trees, permutation importance for all models.

SHAP values describe how the *model* moved its prediction away from its baseline for a
patient. They are associational, not causal. For the FT-Transformer we use permutation
importance (model-agnostic) and do not interpret attention weights as importance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import mae


def _aggregate_onehot(values: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    base = [n.split("=")[0] for n in names]
    feats = list(dict.fromkeys(base))
    agg = np.stack([values[:, [i for i, b in enumerate(base) if b == f]].sum(axis=1) for f in feats], axis=1)
    return agg, feats


def shap_summary(model, df: pd.DataFrame, n: int = 2000, seed: int = 0, points_per_feature: int = 250) -> dict:
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    if model.key == "catboost":
        sv = model.shap_values(sample)
        values, base_value, feats = sv[:, :-1], float(sv[0, -1]), model.feature_names()
        scale = "days"
    else:  # xgboost family: verify with the shap package, aggregate one-hot columns
        import shap
        import xgboost as xgb
        X = model.encoder.one_hot(model._prep(sample))
        expl = shap.TreeExplainer(model.booster)
        raw = expl.shap_values(xgb.DMatrix(X, feature_names=model.feature_names()))
        values, feats = _aggregate_onehot(np.asarray(raw), model.feature_names())
        base_value = float(np.atleast_1d(expl.expected_value)[0])
        scale = "log1p(days)" if model.params.get("target_transform") == "log1p" else "days"
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)
    rng = np.random.default_rng(seed)
    dist = {}
    for j in order:
        f = feats[j]
        idx = rng.choice(len(sample), size=min(points_per_feature, len(sample)), replace=False)
        dist[f] = [{"shap": float(values[i, j]), "value": str(sample.iloc[i][f]) if f in sample else None}
                   for i in idx]
    return {"model": model.key, "scale": scale, "base_value": base_value, "n": int(len(sample)),
            "mean_abs_shap": {feats[j]: float(mean_abs[j]) for j in order},
            "max_abs_shap": float(np.abs(values).max()), "distribution": dist}


def permutation_importance(model, df: pd.DataFrame, columns: list[str], repeats: int = 5, seed: int = 0,
                           n: int = 4000) -> dict:
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    y = sample["Length_of_Stay"].to_numpy()
    base = mae(y, model.predict(sample))
    rng = np.random.default_rng(seed)
    out = {}
    for c in columns:
        incs = []
        for _ in range(repeats):
            shuffled = sample.copy()
            shuffled[c] = rng.permutation(shuffled[c].to_numpy())
            incs.append(mae(y, model.predict(shuffled)) - base)
        out[c] = {"mae_increase": float(np.mean(incs)), "sd": float(np.std(incs))}
    return {"model": model.key, "baseline_mae": base, "importance": out}
