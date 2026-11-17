"""Lag and rolling-window features along the tunnel chainage order."""

import logging

import pandas as pd

from . import register_feature_builder

logger = logging.getLogger(__name__)

DEFAULT_COLUMNS = ["rpm_rotate_per_minute", "chd_knm", "thrust_kn", "ucs_mpa"]
DEFAULT_LAGS = [1, 3]
DEFAULT_WINDOWS = [3, 5]


@register_feature_builder("temporal")
def add_temporal_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Add lag and rolling-mean features of input variables (target excluded).

    Rows are assumed to be ordered along the tunnel chainage. Only past values
    are used (shift/rolling look backwards), so these features are causal.

    Args:
        df: Input frame ordered by chainage.
        params: Optional overrides: ``columns``, ``lags``, ``windows``.

    Returns:
        Copy of ``df`` with lag/rolling columns; NaNs at the series head are
        backfilled from the first valid observation.
    """
    params = params or {}
    columns = params.get("columns", DEFAULT_COLUMNS)
    lags = params.get("lags", DEFAULT_LAGS)
    windows = params.get("windows", DEFAULT_WINDOWS)

    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for window in windows:
            df[f"{col}_roll_mean{window}"] = df[col].rolling(window=window, min_periods=1).mean()

    df = df.ffill().bfill()
    logger.info("Temporal features added; columns now: %d", len(df.columns))
    return df
