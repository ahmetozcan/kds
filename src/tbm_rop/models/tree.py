"""Tree-ensemble model constructors (RandomForest, LightGBM, CatBoost)."""

from typing import Any

import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

from . import register_model

DEFAULT_RANDOM_STATE = 42


@register_model("random_forest")
def build_random_forest(params: dict[str, Any]) -> RandomForestRegressor:
    """RandomForest with sane defaults; params override everything."""
    defaults: dict[str, Any] = {
        "n_estimators": 100,
        "random_state": DEFAULT_RANDOM_STATE,
        "n_jobs": -1,
    }
    return RandomForestRegressor(**{**defaults, **params})


@register_model("lightgbm")
def build_lightgbm(params: dict[str, Any]) -> lgb.LGBMRegressor:
    """LightGBM regressor with quiet logging."""
    defaults: dict[str, Any] = {
        "random_state": DEFAULT_RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    }
    return lgb.LGBMRegressor(**{**defaults, **params})


@register_model("catboost")
def build_catboost(params: dict[str, Any]) -> cb.CatBoostRegressor:
    """CatBoost regressor with quiet logging."""
    defaults: dict[str, Any] = {
        "iterations": 1000,
        "random_state": DEFAULT_RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
    }
    return cb.CatBoostRegressor(**{**defaults, **params})
