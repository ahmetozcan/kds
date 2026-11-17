"""Optuna hyperparameter tuning with nested cross-validation.

The original pipeline selected hyperparameters and reported final scores on
the *same* 5-fold CV (selection bias). This module supports two protocols:

- ``tune_on_folds``: flat search on a fixed fold set — used inside the nested
  loop and to produce a single consensus parameter set for the stacking stage
  (its best value is optimistically biased and must be labelled as such).
- ``nested_oof_predictions``: honest protocol. For each outer fold, a fresh
  Optuna study runs on inner folds built from the outer-train subset only;
  the best configuration is refit on the outer-train and evaluated on the
  outer-test rows it never saw during selection.

Search spaces mirror the original ``7OptunaMultiUKF.py`` ranges.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import r2_score

from .cv import CVConfig, make_folds
from .models import create_model

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

SpaceFn = Callable[[optuna.Trial], dict[str, Any]]

SEARCH_SPACE_REGISTRY: dict[str, SpaceFn] = {}


def register_search_space(name: str) -> Callable[[SpaceFn], SpaceFn]:
    """Register an Optuna search-space function under ``name``."""

    def decorator(fn: SpaceFn) -> SpaceFn:
        SEARCH_SPACE_REGISTRY[name] = fn
        return fn

    return decorator


@register_search_space("random_forest")
def _rf_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


@register_search_space("lightgbm")
def _lgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0),
    }


@register_search_space("catboost")
def _catboost_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
    }


@register_search_space("mlp")
def _mlp_space(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }
    for i in range(params["n_layers"]):
        params[f"n_units_l{i}"] = trial.suggest_int(f"n_units_l{i}", 32, 256, log=True)
    return params


@dataclass(frozen=True)
class TuningConfig:
    """Configuration for an Optuna search.

    Attributes:
        n_trials: Trials per study.
        inner_splits: Inner-CV folds inside each outer-train (nested protocol).
        seed: Sampler seed (studies are made reproducible with TPE(seed)).
    """

    n_trials: int = 30
    inner_splits: int = 3
    seed: int = 42


@dataclass(frozen=True)
class NestedResult:
    """Outcome of nested tuning for one model."""

    oof: np.ndarray
    fold_params: list[dict[str, Any]] = field(default_factory=list)
    fold_inner_scores: list[float] = field(default_factory=list)


def _get_space(model_name: str, space_fn: SpaceFn | None) -> SpaceFn:
    if space_fn is not None:
        return space_fn
    if model_name not in SEARCH_SPACE_REGISTRY:
        raise KeyError(f"No search space registered for model {model_name!r}.")
    return SEARCH_SPACE_REGISTRY[model_name]


def _cv_score(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    params: dict[str, Any],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Mean R² of ``model_name(params)`` over the given folds."""
    scores = []
    for train_idx, val_idx in folds:
        model = create_model(model_name, params)
        model.fit(X[train_idx], y[train_idx])
        scores.append(r2_score(y[val_idx], model.predict(X[val_idx])))
    return float(np.mean(scores))


def tune_on_folds(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cfg: TuningConfig,
    space_fn: SpaceFn | None = None,
) -> tuple[dict[str, Any], float]:
    """Run one Optuna study maximizing mean fold R² on a fixed fold set.

    Returns:
        Tuple of (best params, best mean R²). The best value is biased when
        the same folds are also used for final reporting.
    """
    space = _get_space(model_name, space_fn)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    def objective(trial: optuna.Trial) -> float:
        return _cv_score(X, y, model_name, space(trial), folds)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
    logger.info("%s flat search done: best R2=%.4f", model_name, study.best_value)
    return dict(study.best_params), float(study.best_value)


def nested_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    cv_cfg: CVConfig,
    cfg: TuningConfig,
    space_fn: SpaceFn | None = None,
) -> NestedResult:
    """Nested-CV tuning: honest OOF predictions with per-fold model selection.

    For each outer fold, hyperparameters are selected on inner folds carved
    from the outer-train subset (same CV scheme), then the best model is refit
    on the full outer-train and used to predict the outer-test rows.

    Known limitation (blocked scheme): inner folds are cut in local-position
    space of the outer-train subset, which is non-contiguous in chainage after
    the outer-test block is excised. Inner selection may therefore be mildly
    optimistic under ``scheme="blocked"``; the outer metric itself stays
    honest (outer-test rows never enter selection or refit).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    oof = np.full(len(y), np.nan)
    fold_params: list[dict[str, Any]] = []
    fold_inner_scores: list[float] = []

    for i, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        inner_cfg = CVConfig(
            scheme=cv_cfg.scheme,
            n_splits=cfg.inner_splits,
            random_state=cv_cfg.random_state,
            gap=cv_cfg.gap,
        )
        inner_folds = make_folds(len(train_idx), inner_cfg)
        best_params, best_inner = tune_on_folds(
            X[train_idx], y[train_idx], model_name, inner_folds, cfg, space_fn=space_fn
        )

        model = create_model(model_name, best_params)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X[test_idx])

        fold_params.append(best_params)
        fold_inner_scores.append(best_inner)
        logger.info(
            "%s outer fold %d/%d: inner best R2=%.4f, params=%s",
            model_name, i, len(outer_folds), best_inner, best_params,
        )

    return NestedResult(oof=oof, fold_params=fold_params, fold_inner_scores=fold_inner_scores)


def merge_model_params(
    base: dict[str, dict[str, Any]], tuned: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Overlay tuned params onto base model params without mutating inputs.

    Models absent from ``tuned`` keep their base params; tuned keys win on
    conflict. Tuned models absent from ``base`` are ignored — tuning must
    never silently expand the configured ensemble.
    """
    merged = {name: dict(params) for name, params in base.items()}
    for name, params in tuned.items():
        if name in merged:
            merged[name].update(params)
        else:
            logger.warning("Ignoring tuned params for %r: not in configured models.", name)
    return merged


def load_tuned_params(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a ``{model_name: params}`` mapping written by the tune stage."""
    with open(path) as f:
        return json.load(f)
