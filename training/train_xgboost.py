"""Model 2 - XGBoost.

Two variants:
* ``XGBoostLog``: XGBRegressor trained on log(1 + LOS); predictions are back-transformed
  with expm1. The back-transform of a mean on the log scale estimates something closer to
  the conditional *median* than the mean, which suits our primary metric (MAE).
* ``XGBoostAFT``: XGBoost accelerated-failure-time survival model (``survival:aft``).
  Every patient in this dataset was discharged, so all observations are uncensored events
  (lower bound == upper bound == LOS). It yields a full predictive distribution, enabling
  survival metrics (C-index, integrated Brier score, NLL).
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from scipy import stats

from backend.app.ml.features import FEATURE_SUBSETS_A, TabularEncoder, feature_columns
from .base import LOSModel, early_stop_split

MAX_ROUNDS = 2000
ES_ROUNDS = 50


def _space(trial):
    return {
        "feature_subset": trial.suggest_categorical("feature_subset", list(FEATURE_SUBSETS_A)),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
    }


_DEFAULTS = {"feature_subset": "A_full", "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 5.0, "gamma": 1e-3, "reg_alpha": 1e-3,
             "reg_lambda": 1.0}


class _XGBBase(LOSModel):
    family = "tree"
    uses_early_stopping = True

    def default_params(self):
        return dict(_DEFAULTS)

    def suggest(self, trial):
        return _space(trial)

    def _encode_fit(self, df):
        cats, nums = feature_columns(self.feature_set, self.params.get("feature_subset", "A_full"))
        self.encoder = TabularEncoder(cats, nums).fit(df)
        return self.encoder.one_hot(df)

    def _booster_params(self):
        p = self.params
        return {"max_depth": int(p["max_depth"]), "eta": p["learning_rate"], "subsample": p["subsample"],
                "colsample_bytree": p["colsample_bytree"], "min_child_weight": p["min_child_weight"],
                "gamma": p["gamma"], "alpha": p["reg_alpha"], "lambda": p["reg_lambda"],
                "tree_method": "hist", "seed": self.seed, "nthread": self.threads}

    def best_iteration(self):
        return int(self.booster.best_iteration) + 1

    def feature_names(self):
        return self.encoder.one_hot_names()


class XGBoostLog(_XGBBase):
    """Standard XGBRegressor. Target transform (log1p vs none) and objective are
    preprocessing/modelling decisions made inside the inner CV loop."""
    key = "xgboost"
    display_name = "XGBoost"
    complexity = 3

    def default_params(self):
        return {**_DEFAULTS, "target_transform": "log1p", "objective": "reg:squarederror"}

    def suggest(self, trial):
        p = _space(trial)
        p["target_transform"] = trial.suggest_categorical("target_transform", ["log1p", "none"])
        p["objective"] = trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror"])
        return p

    def _fwd(self, y):
        return np.log1p(y) if self.params["target_transform"] == "log1p" else y

    def _inv(self, z):
        return np.expm1(z) if self.params["target_transform"] == "log1p" else z

    def fit(self, df, params=None):
        self.params = {**self.default_params(), **(params or {})}
        df = self._prep_fit(df)
        X = self._encode_fit(df)
        y = self._fwd(self.y(df))
        tr, es = early_stop_split(len(df), self.seed)
        dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=self.feature_names())
        des = xgb.DMatrix(X[es], label=y[es], feature_names=self.feature_names())
        bp = {**self._booster_params(), "objective": self.params["objective"], "eval_metric": "mae"}
        self.booster = xgb.train(bp, dtr, MAX_ROUNDS, evals=[(des, "es")],
                                 early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
        return self

    def _dm(self, df):
        return xgb.DMatrix(self.encoder.one_hot(self._prep(df)), feature_names=self.feature_names())

    def predict(self, df):
        it = (0, self.best_iteration())
        return self._inv(self.booster.predict(self._dm(df), iteration_range=it))

    def shap_values(self, df):
        """TreeSHAP contributions on the model's target scale (last column = bias)."""
        return self.booster.predict(self._dm(df), pred_contribs=True,
                                    iteration_range=(0, self.best_iteration()))


_Z = {"normal": stats.norm, "logistic": stats.logistic}


def _aft_dist(name):
    if name == "extreme":
        # xgboost's "extreme" = minimum extreme value (Gumbel-min): F(z) = 1 - exp(-e^z)
        return stats.gumbel_l
    return _Z[name]


class XGBoostAFT(_XGBBase):
    key = "xgboost_aft"
    display_name = "XGBoost AFT (survival)"
    complexity = 3

    def default_params(self):
        return {**_DEFAULTS, "aft_distribution": "normal", "aft_scale": 1.0}

    def suggest(self, trial):
        p = _space(trial)
        p["aft_distribution"] = trial.suggest_categorical("aft_distribution", ["normal", "logistic", "extreme"])
        p["aft_scale"] = trial.suggest_float("aft_scale", 0.3, 2.0, log=True)
        return p

    def fit(self, df, params=None):
        self.params = {**self.default_params(), **(params or {})}
        df = self._prep_fit(df)
        X = self._encode_fit(df)
        y = self.y(df)
        tr, es = early_stop_split(len(df), self.seed)

        def dm(idx):
            d = xgb.DMatrix(X[idx], feature_names=self.feature_names())
            d.set_float_info("label_lower_bound", y[idx])
            d.set_float_info("label_upper_bound", y[idx])  # uncensored: every stay ended
            return d

        bp = {**self._booster_params(), "objective": "survival:aft", "eval_metric": "aft-nloglik",
              "aft_loss_distribution": self.params["aft_distribution"],
              "aft_loss_distribution_scale": self.params["aft_scale"]}
        self.booster = xgb.train(bp, dm(tr), MAX_ROUNDS, evals=[(dm(es), "es")],
                                 early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
        return self

    def location(self, df) -> np.ndarray:
        """mu in log T = mu + sigma * Z."""
        d = xgb.DMatrix(self.encoder.one_hot(self._prep(df)), feature_names=self.feature_names())
        return np.log(self.booster.predict(d, iteration_range=(0, self.best_iteration())))

    def predict(self, df):
        """Predicted median LOS (minimises expected absolute error)."""
        z = _aft_dist(self.params["aft_distribution"])
        return np.exp(self.location(df) + self.params["aft_scale"] * z.ppf(0.5))

    def survival_function(self, df, times: np.ndarray) -> np.ndarray:
        z = _aft_dist(self.params["aft_distribution"])
        mu = self.location(df)[:, None]
        return z.sf((np.log(times)[None, :] - mu) / self.params["aft_scale"])

    def log_density(self, df, t: np.ndarray) -> np.ndarray:
        z = _aft_dist(self.params["aft_distribution"])
        s = self.params["aft_scale"]
        zz = (np.log(t) - self.location(df)) / s
        return z.logpdf(zz) - np.log(s * t)
