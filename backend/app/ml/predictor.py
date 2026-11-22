"""Serve LOS predictions, conformal intervals and per-patient explanations.

Only inference-friendly artefacts from ``models/`` are loaded (JSON / native CatBoost and
XGBoost files / NumPy weights), so the API does not need PyTorch, scikit-learn or SHAP.
TreeSHAP values come from CatBoost's and XGBoost's native implementations.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .features import CATEGORY_LEVELS, TabularEncoder
from .ft_numpy import FTTransformerNumpy
from .glm_design import GLMDesign

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.environ.get("RECOVERYAI_MODEL_DIR", ROOT / "models"))
REPORT_PATH = Path(os.environ.get("RECOVERYAI_REPORT", ROOT / "validation" / "results" / "validation_report.json"))

FEATURE_LABELS = {
    "Age": "Age", "Gender": "Gender", "Blood Type": "Blood type", "Medical Condition": "Medical condition",
    "Admission Type": "Admission type", "Medication": "Medication", "Test Results": "Test result",
}


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.dir = Path(model_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.entries = self.manifest["models"]
        self._load()

    # ---------------------------------------------------------------- loading
    def _load(self):
        e = self.entries
        self.glm = None
        if "gamma_glm" in e:
            g = e["gamma_glm"]["glm"]
            self.glm = {"design": GLMDesign.from_dict(g["design"]), "coef": np.array(g["coef"]),
                        "shape": g["shape"], "median_factor": g["median_factor"], "point": g["point"],
                        "names": g["column_names"]}
        self.xgb = {}
        for k in ("xgboost", "xgboost_aft"):
            if k in e:
                import xgboost as xgb
                b = xgb.Booster()
                b.load_model(str(self.dir / e[k]["file"]))
                self.xgb[k] = (b, TabularEncoder.from_dict(e[k]["encoder"]))
        self.cat = None
        if "catboost" in e:
            from catboost import CatBoostRegressor
            self.cat = CatBoostRegressor()
            self.cat.load_model(str(self.dir / e["catboost"]["file"]))
        self.ft = None
        if "ft_transformer" in e:
            meta = e["ft_transformer"]["ft"]
            self.ft = (FTTransformerNumpy.load(str(self.dir / e["ft_transformer"]["file"]), meta),
                       TabularEncoder.from_dict(meta["encoder"]), meta)

    # ---------------------------------------------------------------- predictions
    def _frame(self, p: dict) -> pd.DataFrame:
        return pd.DataFrame([{
            "Age": float(p["age"]), "Gender": p["gender"], "Blood Type": p["blood_type"],
            "Medical Condition": p["medical_condition"], "Admission Type": p["admission_type"],
            "Medication": p["medication"], "Test Results": p["test_result"],
        }])

    def _glm_mu(self, df):
        return float(np.exp(self.glm["design"].matrix(df) @ self.glm["coef"])[0])

    def predict_one(self, key: str, df: pd.DataFrame) -> float:
        e = self.entries[key]
        if key == "median_baseline":
            return float(e["value"])
        if key == "condition_median":
            return float(e["table"].get(df["Medical Condition"].iloc[0], e["global"]))
        if key == "gamma_glm":
            mu = self._glm_mu(df)
            return mu * self.glm["median_factor"] if self.glm["point"] == "median" else mu
        if key in self.xgb:
            import xgboost as xgb
            b, enc = self.xgb[key]
            dm = xgb.DMatrix(enc.one_hot(df), feature_names=e["feature_names"])
            out = float(b.predict(dm, iteration_range=(0, e["best_iteration"]))[0])
            if key == "xgboost":
                return math.expm1(out) if e["params"].get("target_transform") == "log1p" else out
            # AFT: booster returns exp(mu); convert to the distribution median
            dist, scale = e["params"]["aft_distribution"], e["params"]["aft_scale"]
            z = {"normal": stats.norm, "logistic": stats.logistic, "extreme": stats.gumbel_l}[dist]
            return float(out * math.exp(scale * z.ppf(0.5)))
        if key == "catboost":
            return float(self.cat.predict(self._cat_pool(df))[0])
        if key == "ft_transformer":
            net, enc, meta = self.ft
            z = net.predict_standardized(enc.codes(df), enc.numeric_matrix(df).astype(np.float64))
            return float(z[0] * meta["y_std"] + meta["y_mean"])
        raise KeyError(key)

    def _cat_pool(self, df):
        from catboost import Pool
        e = self.entries["catboost"]
        X = df[e["feature_names"]].copy()
        for c in e["categorical"]:
            X[c] = X[c].astype(str)
        return Pool(X, cat_features=e["categorical"])

    # ---------------------------------------------------------------- explanations
    def explain(self, key: str, df: pd.DataFrame, pred: float) -> dict[str, Any]:
        if key == "catboost":
            sv = self.cat.get_feature_importance(self._cat_pool(df), type="ShapValues")[0]
            names = self.entries["catboost"]["feature_names"]
            contribs = [{"feature": n, "value": str(df[n].iloc[0]), "days": float(sv[i])} for i, n in enumerate(names)]
            return self._text("SHAP (TreeSHAP)", float(sv[-1]), contribs, "days")
        if key == "xgboost":
            import xgboost as xgb
            b, enc = self.xgb[key]
            e = self.entries[key]
            dm = xgb.DMatrix(enc.one_hot(df), feature_names=e["feature_names"])
            c = b.predict(dm, pred_contribs=True, iteration_range=(0, e["best_iteration"]))[0]
            base = [n.split("=")[0] for n in e["feature_names"]]
            feats = list(dict.fromkeys(base))
            log_scale = e["params"].get("target_transform") == "log1p"
            contribs = []
            for f in feats:
                v = float(sum(c[i] for i, bb in enumerate(base) if bb == f))
                # On the log1p scale a contribution v multiplies (1 + LOS) by exp(v); in days,
                # removing it would change the estimate by (1 + LOS) * (1 - exp(-v)).
                days = (1 - math.exp(-v)) * (pred + 1) if log_scale else v
                contribs.append({"feature": f, "value": str(df[f].iloc[0]), "days": days, "raw": v})
            bias = float(c[-1])
            return self._text("SHAP (TreeSHAP)", math.expm1(bias) if log_scale else bias, contribs, "days")
        if key == "gamma_glm":
            g = self.glm
            x = g["design"].matrix(df)[0]
            contribs = []
            # Group dummy columns back to their factor. Effects are relative to the reference level.
            groups: dict[str, float] = {}
            for name, coef, xv in zip(g["names"], g["coef"], x):
                if name == "Intercept":
                    continue
                f = "Age" if name.startswith("Age") else name.split("[")[0]
                groups[f] = groups.get(f, 0.0) + float(coef * xv)
            for f, lin in groups.items():
                contribs.append({"feature": f, "value": str(df[f].iloc[0]) if f in df else "",
                                 "days": (math.exp(lin) - 1) * pred / math.exp(lin) if lin else 0.0,
                                 "multiplier": math.exp(lin)})
            return self._text("GLM coefficients (relative to reference levels)", None, contribs, "days")
        return {"method": "none", "note": ("Per-patient attributions are not computed for this model. "
                                           "See the Model Validation page for its permutation importance."),
                "contributions": [], "sentences": []}

    @staticmethod
    def _text(method, base, contribs, unit):
        contribs = sorted(contribs, key=lambda c: -abs(c["days"]))
        sentences = []
        for c in contribs[:4]:
            label = FEATURE_LABELS.get(c["feature"], c["feature"])
            direction = "increased" if c["days"] > 0 else "decreased"
            if abs(c["days"]) < 0.005:
                sentences.append(f"{label} ({c['value']}) had essentially no effect on this model's estimate.")
            else:
                sentences.append(f"{label} ({c['value']}) {direction} the model's estimate by about "
                                 f"{abs(c['days']):.2f} days relative to its baseline.")
        total = sum(abs(c["days"]) for c in contribs)
        return {"method": method, "base_value": base, "unit": unit, "contributions": contribs,
                "sentences": sentences, "total_abs_days": total,
                "caveat": ("These attributions describe how the model arrived at its number. They are "
                           "associations learned from synthetic data, not causes.")}

    # ---------------------------------------------------------------- public API
    def predict(self, p: dict, model: str | None = None, level: float = 0.9) -> dict[str, Any]:
        df = self._frame(p)
        days_in = max(0.0, float(p.get("days_hospitalized") or 0.0))
        level_key = min(self.entries[next(iter(self.entries))]["conformal_q"], key=lambda k: abs(float(k) - level))
        key = model or self.manifest["display_model"]
        results = {}
        for k, e in self.entries.items():
            pr = self.predict_one(k, df)
            q = e["conformal_q"][level_key]
            lo, hi = max(0.0, pr - q), pr + q
            results[k] = {"name": e["display_name"], "total_los": pr, "interval": [lo, hi],
                          "remaining": max(0.0, pr - days_in),
                          "remaining_interval": [max(0.0, lo - days_in), max(0.0, hi - days_in)],
                          "test_mae": e["test_metrics"]["mae"]}
        out = {"model": key, "level": float(level_key), "days_hospitalized": days_in,
               "prediction": results[key], "all_models": results,
               "explanation": self.explain(key, df, results[key]["total_los"]),
               "quality_gate": self.manifest["quality_gate"],
               "selected_model": self.manifest["selected_model"],
               "conditional": self.conditional_remaining(df, days_in, float(level_key))}
        if days_in > results[key]["total_los"]:
            out["note"] = ("You have already been in hospital longer than the model's estimated total stay, so the "
                           "clamped remaining estimate is 0 days. See the survival-adjusted estimate, which accounts "
                           "for time already spent.")
        return out

    def conditional_remaining(self, df, d: float, level: float) -> dict[str, Any] | None:
        """Survival-adjusted remaining stay from the Gamma GLM: distribution of T - d given T > d."""
        if not self.glm:
            return None
        mu, k = self._glm_mu(df), self.glm["shape"]
        dist = stats.gamma(a=k, scale=mu / k)
        f_d = float(dist.cdf(d))
        if f_d > 0.999:
            return {"available": False, "reason": "Elapsed stay is beyond the range the model can describe."}
        q = lambda pr: float(dist.ppf(f_d + pr * (1 - f_d)) - d)
        a = (1 - level) / 2
        return {"available": True, "median_remaining": q(0.5), "interval": [q(a), q(1 - a)],
                "prob_still_admitted_now": 1 - f_d, "method": "Gamma GLM, conditioned on stay > days already spent"}

    def options(self) -> dict[str, Any]:
        return {"categories": {k: v for k, v in CATEGORY_LEVELS.items() if k != "Insurance Provider"},
                "age_range": [18, 90], "models": {k: e["display_name"] for k, e in self.entries.items()},
                "display_model": self.manifest["display_model"], "selected_model": self.manifest["selected_model"],
                "levels": sorted(float(x) for x in self.entries["median_baseline"]["conformal_q"]),
                "quality_gate": self.manifest["quality_gate"]}


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


@lru_cache(maxsize=1)
def get_report() -> dict:
    if not REPORT_PATH.exists():
        return {}
    # parse_constant maps any NaN/Infinity to None so the response is valid JSON
    report = json.loads(REPORT_PATH.read_text(), parse_constant=lambda _: None)
    pc = REPORT_PATH.parent / "positive_control.json"
    if pc.exists():
        report["positive_control"] = json.loads(pc.read_text(), parse_constant=lambda _: None)
    return report


def days_between(a: date, b: date) -> int:
    return max(0, (b - a).days)
