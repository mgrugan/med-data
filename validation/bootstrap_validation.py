"""Non-parametric bootstrap of final test-set predictions.

* 95% percentile and BCa intervals for MAE, RMSE, MedAE, R², bias.
* Paired bootstrap of metric *differences* between models: the same resampled patients
  are used for both models in every replicate, so the interval reflects sampling
  variation in the comparison itself rather than in each model separately.

BCa acceleration is estimated with a grouped (delete-a-group) jackknife over G random
groups, which approximates the full jackknife at a fraction of the cost for n ≈ 11k.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def _vec_metrics(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    Y, P = y[idx], p[idx]
    E = P - Y
    sst = ((Y - Y.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    return {
        "mae": np.abs(E).mean(axis=-1),
        "rmse": np.sqrt((E ** 2).mean(axis=-1)),
        "medae": np.median(np.abs(E), axis=-1),
        "r2": 1 - (E ** 2).sum(axis=-1) / sst,
        "bias": E.mean(axis=-1),
    }


def _point(y, p):
    return {k: float(v[0]) for k, v in _vec_metrics(y, p, np.arange(len(y))[None, :]).items()}


def bootstrap_indices(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def _jackknife_groups(n: int, G: int, seed: int) -> list[np.ndarray]:
    perm = np.random.default_rng(seed + 7).permutation(n)
    return [np.setdiff1d(np.arange(n), g, assume_unique=True) for g in np.array_split(perm, G)]


def _bca(theta_hat: float, boot: np.ndarray, jack: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    prop = np.clip(np.mean(boot < theta_hat), 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop)
    jm = jack.mean()
    num, den = np.sum((jm - jack) ** 3), 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = num / den if den > 0 else 0.0
    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    a2 = stats.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    return float(np.quantile(boot, a1)), float(np.quantile(boot, a2))


def _jack_stats(fn: Callable[[np.ndarray], dict], groups) -> dict[str, np.ndarray]:
    rows = [fn(g) for g in groups]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def bootstrap_model(y: np.ndarray, p: np.ndarray, B: int = 1000, seed: int = 0, G: int = 200,
                    idx: np.ndarray | None = None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = bootstrap_indices(len(y), B, seed) if idx is None else idx
    boots = _vec_metrics(y, p, idx)
    point = _point(y, p)
    jack = _jack_stats(lambda g: _point(y[g], p[g]), _jackknife_groups(len(y), G, seed))
    out = {}
    for k, b in boots.items():
        lo, hi = np.percentile(b, [2.5, 97.5])
        blo, bhi = _bca(point[k], b, jack[k])
        out[k] = {"estimate": point[k], "boot_mean": float(b.mean()), "boot_se": float(b.std(ddof=1)),
                  "ci_percentile": [float(lo), float(hi)], "ci_bca": [blo, bhi]}
    return {"B": int(idx.shape[0]), "metrics": out, "mae_samples": boots["mae"][:400].tolist()}


def bootstrap_difference(y, p_a, p_b, B: int = 1000, seed: int = 0, idx: np.ndarray | None = None,
                         G: int = 200) -> dict:
    """Paired bootstrap of metric(A) - metric(B). Negative MAE diff => A has lower error."""
    y, p_a, p_b = (np.asarray(v, float) for v in (y, p_a, p_b))
    idx = bootstrap_indices(len(y), B, seed) if idx is None else idx
    ma, mb = _vec_metrics(y, p_a, idx), _vec_metrics(y, p_b, idx)
    pa, pb = _point(y, p_a), _point(y, p_b)
    groups = _jackknife_groups(len(y), G, seed)
    ja = _jack_stats(lambda g: _point(y[g], p_a[g]), groups)
    jb = _jack_stats(lambda g: _point(y[g], p_b[g]), groups)
    out = {}
    for k in ["mae", "rmse", "r2"]:
        d = ma[k] - mb[k]
        est = pa[k] - pb[k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        blo, bhi = _bca(est, d, ja[k] - jb[k])
        # two-sided bootstrap p-value for H0: difference = 0
        p_two = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
        out[k] = {"difference": est, "ci_percentile": [float(lo), float(hi)], "ci_bca": [blo, bhi],
                  "p_value": p_two, "prob_a_better": float(np.mean(d < 0) if k != "r2" else np.mean(d > 0)),
                  "excludes_zero": bool(lo > 0 or hi < 0)}
    return out
