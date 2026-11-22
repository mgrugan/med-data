"""Regression metrics (all in days, except R² and MAPE)."""
from __future__ import annotations

import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def medae(y, p):
    return float(np.median(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else float("nan")


def mape(y, p):
    """MAPE is well-defined here because LOS >= 1 day for every record."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = y > 0
    return float(np.mean(np.abs((y[ok] - p[ok]) / y[ok])) * 100)


def bias(y, p):
    """Mean prediction bias = mean(prediction - observed). Positive = over-predicts stay."""
    return float(np.mean(np.asarray(p) - np.asarray(y)))


METRICS = {"mae": mae, "rmse": rmse, "medae": medae, "r2": r2, "mape": mape, "bias": bias}


def all_metrics(y, p) -> dict[str, float]:
    return {k: f(y, p) for k, f in METRICS.items()}
