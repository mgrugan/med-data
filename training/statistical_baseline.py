"""Model 0 - Gamma GLM with log link (interpretable statistical baseline).

Why a Gamma GLM: LOS is positive and (in real hospitals) right-skewed with variance that
grows with the mean, which the Gamma family with a log link captures. Coefficients are
interpretable as multiplicative effects on expected LOS: exp(beta).

Information criteria (AIC, AICc, BIC) are computed here because this model has a proper
likelihood. They are used ONLY to compare likelihood-based models fitted to the same
response (e.g. GLM variants and the null model). They are NOT computed for boosted trees
or neural networks: those models have no well-defined parameter count / likelihood in
the sense the criteria assume, so their "AIC" would not be comparable.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from backend.app.ml.features import FEATURE_SUBSETS_A, SET_A_CATEGORICAL, SET_B_EXTRA_CATEGORICAL
from backend.app.ml.glm_design import GLMDesign
from .base import LOSModel


def _gamma_median_factor(shape: float) -> float:
    """median / mean of a Gamma(shape) distribution."""
    return float(stats.gamma.ppf(0.5, a=shape) / shape)


class GammaGLM(LOSModel):
    key = "gamma_glm"
    display_name = "Gamma GLM (log link)"
    family = "statistical"
    complexity = 2
    likelihood_based = True

    def grid(self):
        return {"feature_subset": list(FEATURE_SUBSETS_A), "age_term": ["linear", "quadratic", "binned"],
                "point": ["mean", "median"]}

    def suggest(self, trial):
        g = self.grid()
        return {k: trial.suggest_categorical(k, v) for k, v in g.items()}

    def default_params(self):
        return {"feature_subset": "A_full", "age_term": "linear", "point": "median"}

    def _cats(self, subset: str) -> list[str]:
        if self.feature_set == "B":
            return SET_A_CATEGORICAL + SET_B_EXTRA_CATEGORICAL
        return [c for c in FEATURE_SUBSETS_A[subset] if c in SET_A_CATEGORICAL]

    def fit(self, df, params=None):
        self.params = {**self.default_params(), **(params or {})}
        df = self._prep_fit(df)
        self.design = GLMDesign(self._cats(self.params["feature_subset"]), self.params["age_term"]).fit(df)
        X = self.design.matrix(df)
        if self.feature_set == "B":
            extra = ["Room Number", "admit_month", "Billing Amount", "doctor_freq", "hospital_freq"]
            self.extra_stats = {c: (float(df[c].mean()), float(df[c].std() or 1.0)) for c in extra}
            X = np.column_stack([X] + [(df[c] - m) / s for c, (m, s) in self.extra_stats.items()])
        y = self.y(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log())).fit()
        self.scale_ = float(self.result_.scale)       # dispersion phi; shape = 1/phi
        self.shape_ = 1.0 / self.scale_
        self.median_factor_ = _gamma_median_factor(self.shape_)
        self.n_ = len(y)
        self._y_train = y
        return self

    def _X(self, df):
        df = self._prep(df)
        X = self.design.matrix(df)
        if self.feature_set == "B":
            X = np.column_stack([X] + [(df[c] - m) / s for c, (m, s) in self.extra_stats.items()])
        return X

    def predict_mean(self, df) -> np.ndarray:
        return np.exp(self._X(df) @ self.result_.params)

    def predict(self, df):
        mu = self.predict_mean(df)
        return mu * self.median_factor_ if self.params["point"] == "median" else mu

    # ---- statistical reporting -----------------------------------------------------
    def column_names(self) -> list[str]:
        names = self.design.column_names()
        if self.feature_set == "B":
            names += [f"{c} (standardised)" for c in self.extra_stats]
        return names

    def information_criteria(self) -> dict[str, float]:
        """Exact Gamma log-likelihood at the fitted mean with MLE-consistent dispersion.

        k counts regression coefficients + 1 dispersion parameter.
        """
        llf = float(self.result_.llf)
        k = int(len(self.result_.params) + 1)
        n = self.n_
        aic = -2 * llf + 2 * k
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        bic = -2 * llf + k * np.log(n)
        return {"log_likelihood": llf, "k_parameters": k, "n": n, "aic": aic, "aicc": aicc,
                "bic": float(bic), "deviance": float(self.result_.deviance),
                "pearson_chi2": float(self.result_.pearson_chi2), "dispersion": self.scale_}

    def coefficient_table(self) -> list[dict[str, Any]]:
        r = self.result_
        ci = r.conf_int(alpha=0.05)
        rows = []
        for i, name in enumerate(self.column_names()):
            b, se = float(r.params[i]), float(r.bse[i])
            rows.append({"term": name, "coef": b, "std_err": se, "z": float(r.tvalues[i]),
                         "p_value": float(r.pvalues[i]), "ci_low": float(ci[i, 0]), "ci_high": float(ci[i, 1]),
                         "exp_coef": float(np.exp(b)), "exp_ci_low": float(np.exp(ci[i, 0])),
                         "exp_ci_high": float(np.exp(ci[i, 1]))})
        return rows

    def null_comparison(self) -> dict[str, Any]:
        """Likelihood-ratio test of the full model against an intercept-only Gamma GLM.

        Both models share the same response, family and data, so the LR statistic and
        the AIC/BIC differences are valid comparisons here.
        """
        y = self._y_train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            null = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Gamma(sm.families.links.Log())).fit(
                scale=self.scale_)
            full_ll = float(self.result_.llf)
        # Evaluate both log-likelihoods at a common dispersion for a proper LR test.
        mu_full = self.result_.fittedvalues
        mu_null = null.fittedvalues
        shape = self.shape_
        ll = lambda mu: float(np.sum(stats.gamma.logpdf(y, a=shape, scale=mu / shape)))
        ll_full, ll_null = ll(mu_full), ll(mu_null)
        df_diff = int(len(self.result_.params) - 1)
        lr = 2 * (ll_full - ll_null)
        k_full, k_null = df_diff + 2, 2
        n = len(y)
        return {
            "lr_statistic": lr, "df": df_diff, "p_value": float(stats.chi2.sf(max(lr, 0), df_diff)),
            "null_log_likelihood": ll_null, "full_log_likelihood_common_dispersion": ll_full,
            "full_log_likelihood_reported": full_ll,
            "aic_null": -2 * ll_null + 2 * k_null, "aic_full": -2 * ll_full + 2 * k_full,
            "bic_null": -2 * ll_null + k_null * np.log(n), "bic_full": -2 * ll_full + k_full * np.log(n),
            "interpretation": ("AIC/BIC lower for the null model means the added predictors do not "
                               "improve the penalised likelihood enough to justify their parameters."),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"design": self.design.to_dict(), "coef": self.result_.params.tolist(),
                "shape": self.shape_, "median_factor": self.median_factor_,
                "point": self.params["point"], "params": self.params,
                "column_names": self.column_names()}


def loocv_demo(df: pd.DataFrame, n: int, seed: int) -> dict[str, Any]:
    """Methodological demonstration of LOOCV on a small subsample, GLM only.

    LOOCV on all ~44k development rows would need ~44k refits per model (and per
    hyper-parameter setting). For boosted trees and neural networks that is infeasible and
    offers little over repeated K-fold CV: LOOCV has low bias but high variance, and its
    folds are almost identical so it says little about stability. Here we show it on a
    subsample for the cheap GLM and compare with 10-fold CV on the same rows.
    """
    from sklearn.model_selection import KFold
    sub = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    y = sub["Length_of_Stay"].to_numpy()
    params = {"feature_subset": "A_full", "age_term": "linear", "point": "median"}
    loo_err = np.empty(len(sub))
    for i in range(len(sub)):
        tr = sub.drop(index=i)
        m = GammaGLM(seed=seed).fit(tr, params)
        loo_err[i] = abs(m.predict(sub.iloc[[i]])[0] - y[i])
    kf_err = np.empty(len(sub))
    for tr_idx, va_idx in KFold(10, shuffle=True, random_state=seed).split(sub):
        m = GammaGLM(seed=seed).fit(sub.iloc[tr_idx], params)
        kf_err[va_idx] = np.abs(m.predict(sub.iloc[va_idx]) - y[va_idx])
    med_err = np.empty(len(sub))
    for i in range(len(sub)):
        med_err[i] = abs(np.median(np.delete(y, i)) - y[i])
    return {"n": int(len(sub)), "model_fits_required": int(len(sub)),
            "loocv_mae": float(loo_err.mean()), "loocv_mae_se": float(loo_err.std(ddof=1) / np.sqrt(len(sub))),
            "kfold10_mae": float(kf_err.mean()), "median_baseline_loocv_mae": float(med_err.mean()),
            "full_data_fits_that_would_be_required": int(len(df))}
