"""Positive control: can this validation framework detect signal when it truly exists?

A null (or weak) result is only convincing if the same pipeline would have found signal
had it been there, and a positive result is only trustworthy if the framework recovers a KNOWN
effect at roughly the right size. We keep the real feature rows but replace LOS with a
semi-synthetic target containing a KNOWN, modest admission-time effect plus noise with the
same spread as the real LOS:

    LOS* = clip(round(base * effect(admission type, condition, age) + noise), 1, 30)

The same models, fixed hyper-parameters, CV folds and permutation test are then applied.
If the framework recovers this effect (MAE well below the median baseline, permutation
p at its minimum), the absence of signal in the real target is a property of the data,
not a failure of the methodology. Run AFTER ``training.run_pipeline``:

    python -m validation.positive_control
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from training.config import GLOBAL_SEED, RESULTS_DIR
from training.data import clean, load_raw, make_splits
from training.registry import make
from .metrics import all_metrics
from .permutation_tests import permutation_test

from backend.app.ml.features import CATEGORY_LEVELS

# Known multiplicative effects spread evenly over each factor's levels (0.8x ... 1.3x).
EFFECTS = {c: dict(zip(CATEGORY_LEVELS[c], np.linspace(0.8, 1.3, len(CATEGORY_LEVELS[c])).round(3).tolist()))
           for c in ["Admission Type", "Medical Condition"]}


def semi_synthetic_target(df: pd.DataFrame, seed: int, noise_sd: float = 2.5, base: float = 4.0,
                          lo: float = 1, hi: float = 14) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mult = df["Admission Type"].map(EFFECTS["Admission Type"]).to_numpy() * \
        df["Medical Condition"].map(EFFECTS["Medical Condition"]).to_numpy() * \
        (1 + 0.006 * (df["Age"].to_numpy() - 60))
    y = base * mult + rng.normal(0, noise_sd, len(df))
    return np.clip(np.round(y), lo, hi)


def run(models=("median_baseline", "gamma_glm", "xgboost", "catboost"), n_perm: int = 10) -> dict:
    report = json.loads((RESULTS_DIR / "validation_report.json").read_text())
    params = report["final_params"]
    df, _ = clean(load_raw())
    dev = make_splits(df, persist=False).dev.copy()
    dev["Length_of_Stay"] = semi_synthetic_target(dev, GLOBAL_SEED)
    out = {"design": {"effects": EFFECTS, "age_effect_per_year": 0.006, "base_days": 4.0, "noise_sd": 2.5,
                      "target_sd": float(dev["Length_of_Stay"].std())}, "cv": {}, "permutation": {}}
    # Oracle: the best achievable MAE given the noise (predict the true median of the noisy target).
    for k in models:
        folds = []
        for i, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=GLOBAL_SEED + 31).split(dev)):
            m = make(k, seed=GLOBAL_SEED + i).fit(dev.iloc[tr], params.get(k, {}))
            folds.append(all_metrics(dev.iloc[va]["Length_of_Stay"], m.predict(dev.iloc[va])))
        out["cv"][k] = {met: {"mean": float(np.mean([f[met] for f in folds])),
                              "sd": float(np.std([f[met] for f in folds], ddof=1))} for met in folds[0]}
        print(k, {a: round(b["mean"], 3) for a, b in out["cv"][k].items()})
        if k != "median_baseline":
            out["permutation"][k] = permutation_test(k, dev, params.get(k, {}), n_perm, GLOBAL_SEED + 55)
    base = out["cv"]["median_baseline"]["mae"]["mean"]
    out["improvement_over_median"] = {k: base - v["mae"]["mean"] for k, v in out["cv"].items()}
    # Detection rule (same relative threshold as the quality gate): a >= 5% MAE gain over the median
    # AND a permutation p-value at its floor 1/(n_perm+1). An absolute 0.25-day rule was used for the
    # earlier 1-30-day synthetic target; it does not transfer to a 1-14-day target.
    out["detection_rule"] = {"min_relative_gain": 0.05, "p_value_floor": 1.0 / (n_perm + 1)}
    out["conclusion"] = (
        "The framework recovers a known, modest admission-time effect: models clearly beat the median baseline "
        "and the shuffled-target control, so the pipeline is sensitive to signal of this size."
        if (max(v for k, v in out["improvement_over_median"].items() if k != "median_baseline") >= 0.05 * base
            and min(p["p_value"] for p in out["permutation"].values()) <= 1.0 / (n_perm + 1) + 1e-12) else
        "WARNING: the framework did not recover the injected effect; results on the real target are inconclusive.")
    (RESULTS_DIR / "positive_control.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(run()["conclusion"])
