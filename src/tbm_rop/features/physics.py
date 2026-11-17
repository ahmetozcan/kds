"""Physics-informed feature engineering (stage 1 of the original pipeline)."""

import logging

import numpy as np
import pandas as pd

from . import register_feature_builder

logger = logging.getLogger(__name__)

EPSILON = 1e-6


@register_feature_builder("physics")
def add_physics_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Add interaction, ratio, square, and log features with physical meaning.

    Args:
        df: Input frame with cleaned raw columns.
        params: Unused; present for builder-interface uniformity.

    Returns:
        Copy of ``df`` with 12 additional columns.
    """
    df = df.copy()

    # Power indices: torque x speed and thrust x speed interactions.
    df["power_index_chd_rpm"] = df["chd_knm"] * df["rpm_rotate_per_minute"]
    df["power_index_thrust_rpm"] = df["thrust_kn"] * df["rpm_rotate_per_minute"]

    # Thrust-to-torque ratio.
    df["force_ratio_thrust_chd"] = df["thrust_kn"] / (df["chd_knm"] + EPSILON)

    # Rock mass descriptors.
    df["rock_hardness_abrasivity_index"] = df["ucs_mpa"] * df["cai"]
    df["ucs_weathered_calculated"] = df["ucs_mpa"] / (df["weathering"] + EPSILON)

    # Squares.
    df["thrust_sq"] = df["thrust_kn"] ** 2
    df["rpm_sq"] = df["rpm_rotate_per_minute"] ** 2
    df["ucs_sq"] = df["ucs_mpa"] ** 2

    # Log transforms for right-skewed variables.
    df["log_thrust"] = np.log1p(df["thrust_kn"])
    df["log_chd"] = np.log1p(df["chd_knm"])
    df["log_rpm"] = np.log1p(df["rpm_rotate_per_minute"])
    df["log_watinflow"] = np.log1p(df["watinflow_l_m"])

    logger.info("Physics features added; columns now: %d", len(df.columns))
    return df
