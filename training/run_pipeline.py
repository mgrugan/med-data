"""End-to-end RecoveryAI research pipeline.

    python -m training.run_pipeline --budget full     # committed results
    python -m training.run_pipeline --budget quick    # smoke test (~minutes)

Order of operations (the test set is only touched in stage 8, once, after selection):
  1  load, de-duplicate, describe; create holdout (20%) and conformal-calibration split
  2  nested CV (outer 5 x inner 5; inner 3 for the transformer) for every model
  3  final hyper-parameter search on the full development set (same inner procedure)
  4  repeated 5-fold CV (3 repeats) of final configurations, identical folds for all models
  5  shuffled-target permutation tests, learning curves, LOOCV demonstration
  6  feature set B experiment (CV only)
  7  model selection + statistical comparison -> selection.json is written and LOCKED
  8  fit final models on dev_fit, calibrate conformal intervals on dev_cal, evaluate on the
     untouched test set exactly once: metrics, bootstrap CIs, paired bootstrap differences,
     calibration, subgroup errors, survival metrics, SHAP / permutation importance
  9  quality gate, artefact export, report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.features import (FEATURE_SUBSETS_A, LEAKAGE_RISK_FEATURES, SET_A_CATEGORICAL,
                                     SET_A_NUMERIC, SET_B_EXTRA_CATEGORICAL, SET_B_EXTRA_NUMERIC, TARGET)
from validation import (bootstrap_validation as bv, calibration as cal, conformal_prediction as cp,
                        error_analysis as ea, explainability as ex, learning_curves as lc,
                        model_comparison as mc, survival_metrics as sv)
from validation.metrics import all_metrics
from validation.nested_cv import nested_cv
from validation.permutation_tests import permutation_test
from validation.repeated_cv import repeated_cv
from validation.tuning import tune
from .config import BUDGETS, CONFORMAL_LEVELS, GLOBAL_SEED, MODEL_DIR, RESULTS_DIR, seed_everything
from .data import clean, describe, load_raw, make_splits
from .registry import MODELS, TUNABLE, make
from .statistical_baseline import GammaGLM, loocv_demo

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    raise TypeError(type(o))


def finite(o):
    """Recursively replace NaN / ±inf with None so the JSON is standards-compliant."""
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = json.loads(json.dumps(obj, default=_json_default))  # normalise numpy types first
    path.write_text(json.dumps(finite(obj), indent=1, allow_nan=False))


class Cache:
    def __init__(self, root: Path, budget: str):
        self.dir = root / "cache" / budget
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, name, fn):
        p = self.dir / f"{name}.json"
        if p.exists():
            log(f"cache hit: {name}")
            return json.loads(p.read_text())
        out = fn()
        dump(out, p)
        return json.loads(p.read_text())


def run(budget_name: str) -> dict:
    B = BUDGETS[budget_name]
    seed_everything(GLOBAL_SEED)
    t_start = time.time()
    results_dir = RESULTS_DIR if budget_name == "full" else RESULTS_DIR / budget_name
    model_dir = MODEL_DIR if budget_name == "full" else MODEL_DIR / budget_name
    model_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(results_dir, budget_name)
    kw_all = {"threads": B.threads}
    model_kw = {k: dict(kw_all) for k in B.models}
    model_kw["ft_transformer"] = {**kw_all, "max_epochs": B.ft_max_epochs}

    # ---------------- 1. data ---------------------------------------------------------
    raw = load_raw()
    df, clean_report = clean(raw)
    if B.subsample:
        df = df.sample(n=B.subsample, random_state=GLOBAL_SEED).reset_index(drop=True)
    data_desc = describe(df)
    splits = make_splits(df, persist=budget_name == "full")
    dev, test = splits.dev, splits.test
    dev_fit, dev_cal = dev.iloc[splits.dev_fit_idx], dev.iloc[splits.dev_cal_idx]
    log(f"data: {len(df)} rows -> dev {len(dev)} (fit {len(dev_fit)}, cal {len(dev_cal)}), test {len(test)}")
    test_hash = hashlib.sha256(pd.util.hash_pandas_object(test, index=False).values.tobytes()).hexdigest()

    # ---------------- 2. nested CV ----------------------------------------------------
    nested = {}
    for k in B.models:
        inner = B.inner_folds_nn if MODELS[k].is_neural else B.inner_folds
        log(f"nested CV: {k} (outer {B.outer_folds} x inner {inner}, {B.trials.get(k, 0)} trials)")
        nested[k] = cache.get(f"nested_{k}", lambda k=k, inner=inner: nested_cv(
            k, dev, B.outer_folds, inner, B.trials.get(k, 0), GLOBAL_SEED, model_kw[k], log))

    # ---------------- 3. final tuning on the full development set --------------------
    final_params, final_studies = {}, {}
    for k in B.models:
        if k in TUNABLE:
            inner = B.inner_folds_nn if MODELS[k].is_neural else B.inner_folds
            log(f"final tuning: {k}")
            st = cache.get(f"final_tuning_{k}", lambda k=k, inner=inner: tune(
                k, dev, inner, B.trials[k], GLOBAL_SEED + 7, model_kw[k], context="_final"))
            final_params[k], final_studies[k] = st["best_params"], st
        else:
            final_params[k] = {}

    # ---------------- 4. repeated CV --------------------------------------------------
    repeated = {}
    for k in B.models:
        log(f"repeated CV: {k}")
        repeated[k] = cache.get(f"repeated_{k}", lambda k=k: repeated_cv(
            k, dev, final_params[k], B.outer_folds, B.cv_repeats, GLOBAL_SEED + 31, model_kw[k], log))

    # ---------------- 5. permutation tests, learning curves, LOOCV --------------------
    permutation = {}
    for k in B.models:
        n_perm = B.permutations.get(k, 0)
        if n_perm:
            log(f"permutation test: {k} ({n_perm} shuffles)")
            permutation[k] = cache.get(f"perm_{k}", lambda k=k, n=n_perm: permutation_test(
                k, dev, final_params[k], n, GLOBAL_SEED + 55, model_kw[k], log))
    curves = {}
    for k in ["gamma_glm", "xgboost", "catboost", "ft_transformer"]:
        if k in B.models:
            log(f"learning curve: {k}")
            seeds = 1 if MODELS[k].is_neural else B.learning_curve_seeds
            curves[k] = cache.get(f"lc_{k}", lambda k=k, s=seeds: lc.learning_curve(
                k, dev, final_params[k], s, GLOBAL_SEED + 77, model_kw[k]))
    log("LOOCV demonstration (GLM, subsample)")
    loocv = cache.get("loocv", lambda: loocv_demo(dev, B.loocv_n, GLOBAL_SEED + 99))

    # ---------------- 6. feature set B (research only, CV) ----------------------------
    set_b = {}
    for k in ["gamma_glm", "xgboost", "catboost"]:
        if k in B.models:
            log(f"feature set B repeated CV: {k}")
            pB = {kk: v for kk, v in final_params[k].items() if kk != "feature_subset"}
            set_b[k] = cache.get(f"setB_repeated_{k}", lambda k=k, p=pB: repeated_cv(
                k, dev, p, B.outer_folds, B.cv_repeats, GLOBAL_SEED + 31, {**model_kw[k], "feature_set": "B"}, log))
    n_test_fold = len(dev) // B.outer_folds
    for k, r in set_b.items():
        dA = np.array([f["mae"] for f in repeated[k]["folds"]])
        dB = np.array([f["mae"] for f in r["folds"]])
        r["vs_set_A"] = mc.corrected_resampled_ttest(dB - dA, len(dev) - n_test_fold, n_test_fold)

    # ---------------- 7. selection (LOCKED before the test set is touched) ------------
    complexity = {k: MODELS[k].complexity for k in B.models}
    selection = mc.select_model(nested, repeated, complexity, len(dev), B.outer_folds)
    selection["locked_at"] = datetime.now(timezone.utc).isoformat()
    dump(selection, results_dir / "selection.json")
    log(f"selection locked: selected={selection['selected_model']} display={selection['display_model']}")

    # ---------------- 8. final fit + single test evaluation ---------------------------
    final_models, test_pred, cal_pred, test_metrics = {}, {}, {}, {}
    y_test = test["Length_of_Stay"].to_numpy(float)
    y_cal = dev_cal["Length_of_Stay"].to_numpy(float)
    for k in B.models:
        log(f"final fit: {k}")
        m = make(k, seed=GLOBAL_SEED, **model_kw[k]).fit(dev_fit, final_params[k])
        final_models[k] = m
        cal_pred[k] = m.predict(dev_cal)
        test_pred[k] = m.predict(test)
        test_metrics[k] = all_metrics(y_test, test_pred[k])
        log(f"  TEST {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in test_metrics[k].items()))
    idx = bv.bootstrap_indices(len(test), B.bootstrap, GLOBAL_SEED + 2024)
    boot = {k: bv.bootstrap_model(y_test, test_pred[k], idx=idx, seed=GLOBAL_SEED) for k in B.models}
    diffs = {}
    for i, a in enumerate(B.models):
        for b in B.models[i + 1:]:
            diffs[f"{b}__vs__{a}"] = bv.bootstrap_difference(y_test, test_pred[b], test_pred[a], idx=idx, seed=GLOBAL_SEED)
    # Served intervals use the log-scale (multiplicative) score, pre-specified because LOS is
    # right-skewed; the symmetric absolute-residual version is reported for comparison.
    conformal = {k: cp.evaluate(y_cal, cal_pred[k], y_test, test_pred[k], CONFORMAL_LEVELS, test,
                                dev_fit["Length_of_Stay"].to_numpy(), method="log") for k in B.models}
    conformal_abs = {k: cp.evaluate(y_cal, cal_pred[k], y_test, test_pred[k], CONFORMAL_LEVELS, test,
                                    dev_fit["Length_of_Stay"].to_numpy(), method="absolute") for k in B.models}
    calib = {k: cal.calibration_report(y_test, test_pred[k]) for k in B.models}
    errors = {k: ea.subgroup_errors(test, test_pred[k], seed=GLOBAL_SEED) for k in B.models}
    resid_by_obs = {k: ea.residuals_by_observed(y_test, test_pred[k]) for k in B.models}
    rng = np.random.default_rng(GLOBAL_SEED)
    scatter_idx = rng.choice(len(test), size=min(1200, len(test)), replace=False)
    scatter = {k: {"observed": y_test[scatter_idx].tolist(), "predicted": test_pred[k][scatter_idx].tolist()}
               for k in B.models}

    # survival metrics for the distributional models
    grid = np.arange(1.0, float(df[TARGET].max()))
    survival = {}
    t_fit = dev_fit["Length_of_Stay"].to_numpy(float)
    if "xgboost_aft" in final_models:
        m = final_models["xgboost_aft"]
        survival["xgboost_aft"] = sv.survival_report(y_test, test_pred["xgboost_aft"], m.survival_function(test, grid),
                                                     m.log_density(test, y_test), t_fit, grid)
    if "gamma_glm" in final_models:
        from scipy import stats as st_
        m = final_models["gamma_glm"]
        mu = m.predict_mean(test)
        S = st_.gamma.sf(grid[None, :], a=m.shape_, scale=(mu / m.shape_)[:, None])
        survival["gamma_glm"] = sv.survival_report(y_test, test_pred["gamma_glm"], S,
                                                   st_.gamma.logpdf(y_test, a=m.shape_, scale=mu / m.shape_), t_fit, grid)
    for k in list(survival):
        survival[k]["c_index_note"] = "0.5 = random ordering of stays; 1.0 = perfect ordering."
    point_c_index = {k: sv.concordance_index(y_test, test_pred[k]) for k in B.models
                     if np.std(test_pred[k]) > 0}

    # explainability
    shap = {k: ex.shap_summary(final_models[k], test, seed=GLOBAL_SEED) for k in ["catboost", "xgboost"]
            if k in final_models}
    perm_imp_cols = SET_A_CATEGORICAL + SET_A_NUMERIC
    perm_imp = {k: ex.permutation_importance(final_models[k], test, perm_imp_cols, repeats=3, seed=GLOBAL_SEED,
                                             n=2000 if budget_name == "full" else 500)
                for k in ["gamma_glm", "xgboost", "catboost", "ft_transformer"] if k in final_models}

    # statistical model report (likelihood-based -> AIC/AICc/BIC are meaningful)
    glm = final_models["gamma_glm"]
    glm_alt = []
    for subset in FEATURE_SUBSETS_A:
        for age_term in ["linear", "quadratic", "binned"]:
            g = GammaGLM(seed=GLOBAL_SEED).fit(dev_fit, {"feature_subset": subset, "age_term": age_term,
                                                         "point": "mean"})
            glm_alt.append({"feature_subset": subset, "age_term": age_term, **g.information_criteria()})
    statistical = {"params": final_params["gamma_glm"], "information_criteria": glm.information_criteria(),
                   "coefficients": glm.coefficient_table(), "null_comparison": glm.null_comparison(),
                   "specification_comparison": glm_alt, "loocv_demo": loocv,
                   "note": ("AIC, AICc and BIC compare likelihood-based models fitted to the same response and data. "
                            "They are not reported for boosted trees or neural networks, whose effective number of "
                            "parameters and likelihood are not comparable.")}

    # feature set B final evaluation on test
    for k in set_b:
        pB = {kk: v for kk, v in final_params[k].items() if kk != "feature_subset"}
        mB = make(k, seed=GLOBAL_SEED, **{**model_kw[k], "feature_set": "B"}).fit(dev_fit, pB)
        pr = mB.predict(test)
        set_b[k]["test_metrics"] = all_metrics(y_test, pr)
        set_b[k]["test_diff_vs_A"] = bv.bootstrap_difference(y_test, pr, test_pred[k], idx=idx, seed=GLOBAL_SEED)
        if k == "catboost":
            set_b[k]["shap"] = ex.shap_summary(mB, test, n=1000, seed=GLOBAL_SEED, points_per_feature=50)
        set_b[k].pop("folds", None)

    gate = mc.quality_gate(nested, diffs, permutation, conformal, calib, selection)
    log(f"quality gate: warning={gate['warning']} ({gate['n_failed']} checks failed)")

    # ---------------- 9. export --------------------------------------------------------
    manifest = export_models(final_models, final_params, conformal, test_metrics, selection, gate, model_dir,
                             budget_name, data_desc)
    model_info = {k: {"display_name": MODELS[k].display_name, "family": MODELS[k].family,
                      "complexity": MODELS[k].complexity, "likelihood_based": MODELS[k].likelihood_based,
                      "uses_early_stopping": MODELS[k].uses_early_stopping} for k in B.models}
    for k in nested:
        nested[k].pop("oof_pred", None)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "budget": B.__dict__,
        "runtime_minutes": (time.time() - t_start) / 60,
        "environment": env_info(), "seeds": {"global": GLOBAL_SEED, "holdout_split": GLOBAL_SEED,
                                             "calibration_split": GLOBAL_SEED + 1, "nested_outer": GLOBAL_SEED,
                                             "repeated_cv": [GLOBAL_SEED + 31 + 1000 * r for r in range(B.cv_repeats)],
                                             "bootstrap": GLOBAL_SEED + 2024, "permutation": GLOBAL_SEED + 55},
        "research_questions": {
            "primary": ("To what extent can patient characteristics available near hospital admission predict "
                        "length of stay in a real, de-identified hospital dataset, and how reliably can modern "
                        "machine-learning models quantify uncertainty in those predictions?"),
            "secondary": ("Do advanced nonlinear machine-learning models provide statistically meaningful "
                          "predictive improvement over simple statistical and naive baselines?")},
        "data": {**clean_report, **data_desc, "n_dev": len(dev), "n_dev_fit": len(dev_fit),
                 "n_dev_calibration": len(dev_cal), "n_test": len(test), "test_sha256": test_hash},
        "feature_sets": {"A": {"categorical": SET_A_CATEGORICAL, "numeric": SET_A_NUMERIC,
                               "inner_cv_subsets": FEATURE_SUBSETS_A},
                         "B": {"categorical": SET_A_CATEGORICAL + SET_B_EXTRA_CATEGORICAL,
                               "numeric": SET_A_NUMERIC + SET_B_EXTRA_NUMERIC,
                               "leakage_risk": sorted(LEAKAGE_RISK_FEATURES)},
                         "excluded": {"encounter_id / patient_nbr": "identifiers",
                                      "Race": "not used as a predictor; used only to audit error parity",
                                      "weight": "97% missing", "readmitted": "post-discharge outcome",
                                      "diag_2 / diag_3": "secondary diagnoses are coded at discharge",
                                      "set-B variables": "accumulated during the stay or recorded at discharge"}},
        "models": model_info, "nested_cv": nested, "final_params": final_params,
        "final_tuning": {k: {kk: v for kk, v in s.items() if kk != "trials"} | {"n_trials": len(s["trials"])}
                         for k, s in final_studies.items()},
        "repeated_cv": {k: {kk: v for kk, v in r.items()} for k, r in repeated.items()},
        "permutation_tests": permutation, "learning_curves": curves, "selection": selection,
        "test": {"metrics": test_metrics, "bootstrap": boot, "paired_differences": diffs,
                 "conformal": conformal, "conformal_absolute": conformal_abs, "calibration": calib, "subgroup_errors": errors,
                 "residuals_by_observed": resid_by_obs, "scatter": scatter, "survival": survival,
                 "c_index_point_predictions": point_c_index},
        "explainability": {"shap": shap, "permutation_importance": perm_imp},
        "statistical_model": statistical, "feature_set_b": set_b, "quality_gate": gate,
        "manifest": manifest,
    }
    dump(report, results_dir / "validation_report.json")
    for k, s in final_studies.items():
        dump(s, results_dir / "optuna" / f"{k}_final.json")
    for k, n in nested.items():
        for s in n.get("studies", []):
            dump(s, results_dir / "optuna" / f"{k}{s['context']}.json")
    (results_dir / "pipeline_log.txt").write_text("\n".join(LOG_LINES))
    log(f"done in {(time.time() - t_start) / 60:.1f} min")
    return report


def env_info() -> dict:
    import catboost, optuna, sklearn, statsmodels, torch, xgboost
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "catboost": catboost.__version__, "xgboost": xgboost.__version__,
            "torch": torch.__version__, "optuna": optuna.__version__, "statsmodels": statsmodels.__version__,
            "platform": platform.platform()}


def export_models(models, params, conformal, test_metrics, selection, gate, out: Path, budget, data_desc) -> dict:
    import joblib
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for k, m in models.items():
        e = {"display_name": m.display_name, "family": m.family, "params": params[k],
             "conformal_q": {lv: conformal[k]["levels"][lv]["q"] for lv in conformal[k]["levels"]},
             "conformal_method": conformal[k]["method"],
             "test_metrics": test_metrics[k]}
        if k == "median_baseline":
            e["value"] = m.median_
        elif k == "condition_median":
            e["table"], e["global"] = m.table_, m.global_
        elif k == "gamma_glm":
            e["glm"] = m.to_dict()
        elif k in ("xgboost", "xgboost_aft"):
            m.booster.save_model(str(out / f"{k}.json"))
            e["file"], e["encoder"], e["best_iteration"] = f"{k}.json", m.encoder.to_dict(), m.best_iteration()
            e["feature_names"] = m.feature_names()
        elif k == "catboost":
            m.model.save_model(str(out / "catboost.cbm"))
            e["file"], e["feature_names"] = "catboost.cbm", m.feature_names()
            e["categorical"] = [c for c in m.feature_names() if c in SET_A_CATEGORICAL]
        elif k == "ft_transformer":
            import torch
            torch.save(m.net.state_dict(), out / "ft_transformer.pt")
            e["ft"] = m.export_numpy(out / "ft_transformer.npz")
            e["file"], e["history"] = "ft_transformer.npz", m.history
        entries[k] = e
        if k == "gamma_glm":  # statsmodels keeps the training data; not needed for prediction
            m.result_.remove_data()
            m._y_train = None
        if k != "ft_transformer":
            joblib.dump(m, out / f"{k}.joblib")
    manifest = {"budget": budget, "created_at": datetime.now(timezone.utc).isoformat(), "seed": GLOBAL_SEED,
                "feature_set": "A", "models": entries, "selected_model": selection["selected_model"],
                "display_model": selection["display_model"], "quality_gate": {
                    "warning": gate["warning"], "headline": gate["headline"],
                    "failed": [c["label"] for c in gate["checks"] if c["failed"]]},
                "training_los_median": data_desc["los_median"], "training_los_max": data_desc["los_max"]}
    dump(manifest, out / "manifest.json")
    return {"path": str(out.relative_to(out.parents[1] if budget != "full" else out.parent)),
            "selected_model": manifest["selected_model"], "display_model": manifest["display_model"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="full", choices=list(BUDGETS))
    run(ap.parse_args().budget)
