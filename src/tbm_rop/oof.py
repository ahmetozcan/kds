"""Out-of-fold (OOF) prediction generation for base models."""

import logging
import time

import numpy as np
import pandas as pd

from .cv import Fold
from .models import create_model

logger = logging.getLogger(__name__)


def generate_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    params: dict | None,
    folds: list[Fold],
) -> np.ndarray:
    """Produce OOF predictions for one model across the given folds.

    A fresh estimator is created per fold; each row is predicted by a model
    that never saw it during training.

    Args:
        X: Feature matrix (chainage order preserved).
        y: Target series.
        model_name: Registered model name.
        params: Hyperparameters for the model constructor.
        folds: Train/validation index pairs.

    Returns:
        Array of OOF predictions aligned with ``X`` rows.
    """
    oof = np.full(len(X), np.nan)
    start = time.time()

    for i, (train_idx, val_idx) in enumerate(folds):
        model = create_model(model_name, params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = np.asarray(model.predict(X.iloc[val_idx])).ravel()
        logger.info("  %s fold %d/%d done", model_name, i + 1, len(folds))

    if np.isnan(oof).any():
        n_missing = int(np.isnan(oof).sum())
        logger.warning(
            "%s: %d rows never appeared in a validation fold (blocked CV with gap "
            "does not drop rows, so this indicates a fold construction issue).",
            model_name,
            n_missing,
        )

    logger.info("%s OOF complete in %.1fs", model_name, time.time() - start)
    return oof
