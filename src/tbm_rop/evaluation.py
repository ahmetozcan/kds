"""Metrics and experiment-report helpers."""

import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import collect_environment

logger = logging.getLogger(__name__)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the standard metric bundle used across the project."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save_report(report: dict, path: str | Path) -> Path:
    """Write an experiment report as JSON with a timestamp.

    Args:
        report: Arbitrary JSON-serializable experiment summary.
        path: Output path; parent directories are created.

    Returns:
        The resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **report,
        "environment": collect_environment(),
    }
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Report saved: %s", path)
    return path
