"""Model 1 - CatBoost regression.

CatBoost suits this dataset because almost every predictor is categorical; its ordered
target statistics encode categories without the target leakage of naive target encoding.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor, Pool

from backend.app.ml.features import FEATURE_SUBSETS_A, feature_columns
from .base import LOSModel, early_stop_split


class CatBoostLOS(LOSModel):
    key = "catboost"
    display_name = "CatBoost"
    family = "tree"
    complexity = 4
    uses_early_stopping = True

    def default_params(self):
        return {"feature_subset": "A_full", "depth": 6, "learning_rate": 0.05, "iterations": 1500,
                "l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 1.0,
                "loss_function": "RMSE"}

    def suggest(self, trial):
        return {
            "feature_subset": trial.suggest_categorical("feature_subset", list(FEATURE_SUBSETS_A)),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 300, 2000, step=100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "loss_function": trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
        }

    def _cols(self):
        cats, nums = feature_columns(self.feature_set, self.params["feature_subset"])
        return cats, nums

    def _pool(self, df, y=None):
        cats, nums = self._cols()
        X = df[cats + nums].copy()
        for c in cats:
            X[c] = X[c].astype(str)
        return Pool(X, label=y, cat_features=cats)

    def fit(self, df, params=None):
        self.params = {**self.default_params(), **(params or {})}
        df = self._prep_fit(df)
        y = self.y(df)
        tr, es = early_stop_split(len(df), self.seed)
        p = self.params
        self.model = CatBoostRegressor(
            depth=p["depth"], learning_rate=p["learning_rate"], iterations=p["iterations"],
            l2_leaf_reg=p["l2_leaf_reg"], random_strength=p["random_strength"],
            bagging_temperature=p["bagging_temperature"], loss_function=p["loss_function"],
            eval_metric="MAE", od_type="Iter", od_wait=50, use_best_model=True,
            random_seed=self.seed, thread_count=self.threads, verbose=False, allow_writing_files=False)
        self.model.fit(self._pool(df.iloc[tr], y[tr]), eval_set=self._pool(df.iloc[es], y[es]))
        return self

    def predict(self, df):
        return np.asarray(self.model.predict(self._pool(self._prep(df))), dtype=float)

    def shap_values(self, df):
        """TreeSHAP values in days (last column = expected value)."""
        return self.model.get_feature_importance(self._pool(self._prep(df)), type="ShapValues")

    def feature_names(self):
        cats, nums = self._cols()
        return cats + nums

    def best_iteration(self):
        return int(self.model.get_best_iteration()) + 1
