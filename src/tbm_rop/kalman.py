"""Scalar Kalman filtering/smoothing for denoising model prediction series.

The original pipeline ran ``pykalman.UnscentedKalmanFilter`` with identity
transition and observation functions. For a linear-Gaussian local-level model
the unscented transform is exact, so that construction is mathematically a
plain scalar Kalman filter / RTS smoother. This module implements both:

- ``mode="causal"``: forward filter only. The estimate at row ``t`` uses
  observations up to ``t`` — usable in real-time and free of look-ahead.
- ``mode="smoother"``: forward filter + RTS backward pass. The estimate at
  row ``t`` uses the whole series (non-causal); kept for replicating the
  original pipeline and for ablation.
- ``mode="none"``: identity passthrough. The series is returned unfiltered —
  the ablation control that separates the effect of Kalman filtering from
  the effect of adding OOF predictions as features at all.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KalmanConfig:
    """Configuration for the local-level Kalman denoiser.

    Attributes:
        process_noise: Transition variance Q of the random-walk state.
        observation_noise: Observation variance R (e.g. model MSE).
        initial_state_variance: Variance of the initial state estimate.
        mode: ``"causal"`` (filter), ``"smoother"`` (filter + RTS pass),
            or ``"none"`` (identity passthrough, ablation control).
    """

    process_noise: float
    observation_noise: float
    initial_state_variance: float = 1.0
    mode: str = "causal"


def _forward_filter(
    observations: np.ndarray, cfg: KalmanConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the forward Kalman filter.

    Returns:
        Tuple of (filtered means, filtered variances,
        one-step-ahead predicted means, predicted variances).
    """
    n = len(observations)
    means = np.zeros(n)
    variances = np.zeros(n)
    pred_means = np.zeros(n)
    pred_variances = np.zeros(n)

    mean, var = observations[0], cfg.initial_state_variance
    pred_means[0], pred_variances[0] = mean, var

    gain = var / (var + cfg.observation_noise)
    mean = mean + gain * (observations[0] - mean)
    var = (1.0 - gain) * var
    means[0], variances[0] = mean, var

    for t in range(1, n):
        pred_mean, pred_var = mean, var + cfg.process_noise
        pred_means[t], pred_variances[t] = pred_mean, pred_var

        gain = pred_var / (pred_var + cfg.observation_noise)
        mean = pred_mean + gain * (observations[t] - pred_mean)
        var = (1.0 - gain) * pred_var
        means[t], variances[t] = mean, var

    return means, variances, pred_means, pred_variances


def _rts_smoother(
    means: np.ndarray,
    variances: np.ndarray,
    pred_means: np.ndarray,
    pred_variances: np.ndarray,
    cfg: KalmanConfig,
) -> np.ndarray:
    """Run the Rauch-Tung-Striebel backward pass over filtered estimates."""
    n = len(means)
    smoothed = np.zeros(n)
    smoothed[-1] = means[-1]

    for t in range(n - 2, -1, -1):
        gain = variances[t] / pred_variances[t + 1]
        smoothed[t] = means[t] + gain * (smoothed[t + 1] - pred_means[t + 1])

    return smoothed


def denoise_series(observations: np.ndarray, cfg: KalmanConfig) -> np.ndarray:
    """Denoise a 1-D series with a local-level Kalman filter.

    Args:
        observations: Noisy series (e.g. out-of-fold model predictions ordered
            by chainage).
        cfg: Kalman configuration; ``cfg.mode`` selects causal filtering or
            RTS smoothing.

    Returns:
        Denoised series of the same length.

    Raises:
        ValueError: For an unknown mode or empty input.
    """
    observations = np.asarray(observations, dtype=float).ravel()
    if observations.size == 0:
        raise ValueError("Cannot denoise an empty series.")
    if cfg.mode not in {"causal", "smoother", "none"}:
        raise ValueError(
            f"Unknown Kalman mode: {cfg.mode!r} (use 'causal', 'smoother', or 'none')."
        )
    if cfg.mode == "none":
        return observations.copy()

    means, variances, pred_means, pred_variances = _forward_filter(observations, cfg)
    if cfg.mode == "causal":
        return means

    smoothed = _rts_smoother(means, variances, pred_means, pred_variances, cfg)
    logger.debug("RTS smoothing complete (n=%d)", len(smoothed))
    return smoothed


def denoise_oof_foldwise(
    oof: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    process_noise: float,
    mode: str = "causal",
    initial_state_variance: float = 1.0,
) -> tuple[np.ndarray, list[float]]:
    """Denoise an OOF series using a fold-local observation noise.

    The observation noise ``R`` is the mean squared error of the OOF series,
    but computed on the *training* rows of each fold only. A row's own target
    therefore never influences the filter that produces that row's denoised
    value. The filter still runs over the whole chainage-ordered series, so
    the recursive structure is unchanged; only ``R`` is fold-local, and each
    fold contributes the values of its validation rows.

    At deployment ``R`` would be estimated from all available history, which
    is the single-pass behaviour of :func:`denoise_series`. The fold-local
    estimate exists so that cross-validated metrics carry no target
    information from the rows being scored.

    Args:
        oof: Out-of-fold prediction series in chainage order.
        y: Target series aligned with ``oof``.
        folds: Train/validation index pairs covering every row exactly once.
        process_noise: Transition variance Q of the random-walk state.
        mode: ``"causal"``, ``"smoother"`` or ``"none"``.
        initial_state_variance: Variance of the initial state estimate.

    Returns:
        Tuple of (denoised series, per-fold ``R`` estimates).

    Raises:
        ValueError: When the folds leave any row without a value.
    """
    oof = np.asarray(oof, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    denoised = np.full(len(oof), np.nan)
    r_per_fold: list[float] = []

    for train_idx, val_idx in folds:
        r = float(np.mean((y[train_idx] - oof[train_idx]) ** 2))
        cfg = KalmanConfig(
            process_noise=process_noise,
            observation_noise=r,
            initial_state_variance=initial_state_variance,
            mode=mode,
        )
        denoised[val_idx] = denoise_series(oof, cfg)[val_idx]
        r_per_fold.append(r)

    if np.isnan(denoised).any():
        n_missing = int(np.isnan(denoised).sum())
        raise ValueError(
            f"Fold-wise denoising left {n_missing} rows unassigned; the folds "
            "must cover every row exactly once."
        )

    logger.info(
        "Fold-wise denoising done (mode=%s): R ranges %.4f..%.4f across %d folds",
        mode, min(r_per_fold), max(r_per_fold), len(r_per_fold),
    )
    return denoised, r_per_fold
