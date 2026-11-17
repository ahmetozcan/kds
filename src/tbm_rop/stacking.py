"""Two-level stacking: base-model OOF features and the meta-model tournament."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .cv import Fold
from .models import create_model
from .oof import generate_oof_predictions

logger = logging.getLogger(__name__)


def build_meta_features(
    X: pd.DataFrame,
    y: pd.Series,
    base_models: dict[str, dict | None],
    folds: list[Fold],
) -> pd.DataFrame:
    """Generate the meta-feature matrix from base-model OOF predictions.

    Args:
        X: Base feature matrix.
        y: Target series.
        base_models: Mapping of registered model name -> hyperparameters.
        folds: Shared folds used for every base model.

    Returns:
        DataFrame with one OOF-prediction column per base model.
    """
    meta = {}
    for name, params in base_models.items():
        logger.info("Generating OOF predictions for base model '%s'", name)
        meta[name] = generate_oof_predictions(X, y, name, params, folds)
    return pd.DataFrame(meta, index=X.index)


def run_meta_tournament(
    meta_features: pd.DataFrame,
    y: pd.Series,
    meta_models: dict[str, dict | None],
    folds: list[Fold],
) -> dict[str, float]:
    """Evaluate each candidate meta model on the OOF meta features.

    Args:
        meta_features: Base-model OOF prediction columns.
        y: Target series.
        meta_models: Mapping of registered model name -> hyperparameters.
        folds: Folds for meta-level cross-validation.

    Returns:
        Mapping of meta-model name -> mean validation R^2 across folds.
    """
    results: dict[str, float] = {}
    for name, params in meta_models.items():
        fold_scores = []
        for train_idx, val_idx in folds:
            model = create_model(name, params)
            model.fit(meta_features.iloc[train_idx], y.iloc[train_idx])
            preds = np.asarray(model.predict(meta_features.iloc[val_idx])).ravel()
            fold_scores.append(r2_score(y.iloc[val_idx], preds))
        results[name] = float(np.mean(fold_scores))
        logger.info("Meta model '%s': mean R^2 = %.4f", name, results[name])
    return results
