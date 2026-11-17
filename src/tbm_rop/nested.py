"""Nested evaluation of the Kalman-augmented feature table.

The ``denoise`` and ``stack`` stages share a single fold set. A validation row
never carries its own target, so the reported metric is out-of-fold in the
usual sense. The *training* rows of a fold, however, carry OOF prediction
columns produced by base models whose training subsets included that fold's
validation rows. The level-2 learner can therefore see an echo of the targets
it is about to be scored on.

This module removes that path. Inside each outer fold the augmented table is
rebuilt from scratch using the outer-training rows only:

- training rows receive inner-OOF predictions from models fitted on inner
  splits of the outer-training subset;
- outer-validation rows receive predictions from models refitted on the whole
  outer-training subset;
- the Kalman observation noise is estimated on outer-training rows only.

No model contributing to the augmented table has seen an outer-validation
target, so the resulting metric is a strictly nested estimate of the same
method. The method itself is unchanged; only its evaluation is stricter.

Known limitation (blocked scheme): inner folds are cut in local-position
space of the outer-training subset, which is non-contiguous in chainage once
the outer block is excised. This mirrors ``tuning.nested_oof_predictions``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cv import CVConfig, Fold, make_folds
from .kalman import KalmanConfig, denoise_series
from .models import create_model

logger = logging.getLogger(__name__)

OOF_PREFIX = "rop_kf_"


@dataclass(frozen=True)
class NestedStackResult:
    """Nested out-of-fold predictions for each level-2 (super) model."""

    predictions: dict[str, np.ndarray]
    observation_noise: list[dict[str, float]] = field(default_factory=list)


def _fold_augmented_columns(
    X_base: pd.DataFrame,
    y: pd.Series,
    base_models: dict[str, dict[str, Any]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    inner_folds: list[Fold],
    process_noise: float,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build the denoised OOF columns for one outer fold.

    Args:
        X_base: Engineered feature table without any ``rop_kf_*`` column.
        y: Target series aligned with ``X_base``.
        base_models: Level-1 learners mapped to their hyperparameters.
        train_idx: Outer-training row positions.
        val_idx: Outer-validation row positions.
        inner_folds: Train/validation pairs in ``train_idx``-local positions.
        process_noise: Kalman transition variance Q.
        mode: Kalman mode (``"causal"``, ``"smoother"`` or ``"none"``).

    Returns:
        Tuple of (denoised column frame, per-model observation noise).

    Raises:
        ValueError: When the inner folds leave a training row unpredicted.
    """
    n = len(X_base)
    y_arr = y.to_numpy(dtype=float)
    columns: dict[str, np.ndarray] = {}
    noise: dict[str, float] = {}

    for name, params in base_models.items():
        series = np.full(n, np.nan)

        for inner_train_local, inner_val_local in inner_folds:
            inner_train = train_idx[inner_train_local]
            inner_val = train_idx[inner_val_local]
            model = create_model(name, params)
            model.fit(X_base.iloc[inner_train], y.iloc[inner_train])
            series[inner_val] = np.asarray(model.predict(X_base.iloc[inner_val])).ravel()

        if np.isnan(series[train_idx]).any():
            n_missing = int(np.isnan(series[train_idx]).sum())
            raise ValueError(
                f"Inner folds left {n_missing} outer-training rows unpredicted for {name!r}."
            )

        # Every remaining row is filled by a model refitted on the whole
        # outer-training subset. Under the blocked scheme with a purge gap
        # these are the outer-validation rows plus the purged rows, which
        # belong to neither subset but still need a value because the filter
        # runs over an unbroken chainage-ordered series. None of them took
        # part in the fit, so no target of a scored row is involved.
        remaining = np.flatnonzero(np.isnan(series))
        if remaining.size:
            model = create_model(name, params)
            model.fit(X_base.iloc[train_idx], y.iloc[train_idx])
            series[remaining] = np.asarray(model.predict(X_base.iloc[remaining])).ravel()

        if np.isnan(series).any():
            raise ValueError(f"Augmented series for {name!r} still holds unfilled rows.")

        r = float(np.mean((y_arr[train_idx] - series[train_idx]) ** 2))
        kalman_cfg = KalmanConfig(
            process_noise=process_noise, observation_noise=r, mode=mode
        )
        columns[f"{OOF_PREFIX}{name}"] = denoise_series(series, kalman_cfg)
        noise[name] = r

    return pd.DataFrame(columns, index=X_base.index), noise


def nested_stack_evaluation(
    X_base: pd.DataFrame,
    y: pd.Series,
    base_models: dict[str, dict[str, Any]],
    super_models: dict[str, dict[str, Any]],
    outer_folds: list[Fold],
    cv_cfg: CVConfig,
    inner_splits: int,
    process_noise: float,
    mode: str,
) -> NestedStackResult:
    """Produce strictly nested OOF predictions for every level-2 learner.

    Args:
        X_base: Engineered feature table without any ``rop_kf_*`` column.
        y: Target series aligned with ``X_base``.
        base_models: Level-1 learners that generate the OOF signals.
        super_models: Level-2 learners trained on the augmented table.
        outer_folds: Outer train/validation index pairs.
        cv_cfg: CV configuration, reused for the inner splits.
        inner_splits: Number of inner folds inside each outer-training subset.
        process_noise: Kalman transition variance Q.
        mode: Kalman mode.

    Returns:
        Nested predictions per level-2 learner plus the per-fold noise log.
    """
    predictions = {name: np.full(len(y), np.nan) for name in super_models}
    noise_log: list[dict[str, float]] = []

    for i, (train_idx, val_idx) in enumerate(outer_folds, start=1):
        inner_cfg = CVConfig(
            scheme=cv_cfg.scheme,
            n_splits=inner_splits,
            random_state=cv_cfg.random_state,
            gap=cv_cfg.gap,
        )
        inner_folds = make_folds(len(train_idx), inner_cfg)
        logger.info(
            "=== Nested outer fold %d/%d (train %d, val %d, %d inner splits) ===",
            i, len(outer_folds), len(train_idx), len(val_idx), len(inner_folds),
        )

        kf_columns, noise = _fold_augmented_columns(
            X_base, y, base_models, train_idx, val_idx, inner_folds, process_noise, mode
        )
        augmented = pd.concat([X_base, kf_columns], axis=1)
        noise_log.append(noise)

        for name, params in super_models.items():
            model = create_model(name, params)
            model.fit(augmented.iloc[train_idx], y.iloc[train_idx])
            predictions[name][val_idx] = np.asarray(
                model.predict(augmented.iloc[val_idx])
            ).ravel()
            logger.info("  level-2 '%s' fold %d done", name, i)

    for name, preds in predictions.items():
        if np.isnan(preds).any():
            n_missing = int(np.isnan(preds).sum())
            raise ValueError(f"Nested evaluation left {n_missing} rows unscored for {name!r}.")

    return NestedStackResult(predictions=predictions, observation_noise=noise_log)
