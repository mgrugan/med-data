from __future__ import annotations

from .baselines import ConditionMedianBaseline, MedianBaseline
from .statistical_baseline import GammaGLM
from .train_catboost import CatBoostLOS
from .train_ft_transformer import FTTransformerLOS
from .train_xgboost import XGBoostAFT, XGBoostLog

MODELS = {m.key: m for m in [MedianBaseline, ConditionMedianBaseline, GammaGLM, XGBoostLog, XGBoostAFT,
                              CatBoostLOS, FTTransformerLOS]}
TUNABLE = {"gamma_glm", "xgboost", "xgboost_aft", "catboost", "ft_transformer"}


def make(key: str, **kw):
    return MODELS[key](**kw)
