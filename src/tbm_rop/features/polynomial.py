"""Degree-2 polynomial interaction features over selected engineered columns."""

import logging

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from . import register_feature_builder

logger = logging.getLogger(__name__)

DEFAULT_COLUMNS = [
    "power_index_thrust_rpm",
    "ucs_weathered_calculated",
    "log_thrust",
    "weathering",
    "rpm_sq",
]


@register_feature_builder("polynomial")
def add_polynomial_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Expand selected columns into degree-2 polynomial/interaction terms.

    The source columns are dropped and replaced by the polynomial expansion
    (which includes the originals as degree-1 terms), mirroring the original
    pipeline.

    Args:
        df: Input frame containing the columns to expand.
        params: Optional overrides: ``columns``, ``degree``.

    Returns:
        Copy of ``df`` with polynomial columns appended.
    """
    params = params or {}
    columns = [c for c in params.get("columns", DEFAULT_COLUMNS) if c in df.columns]
    degree = params.get("degree", 2)

    df = df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    expanded = poly.fit_transform(df[columns])
    poly_df = pd.DataFrame(
        expanded, columns=poly.get_feature_names_out(columns), index=df.index
    )

    df = df.drop(columns=columns)
    df = pd.concat([df, poly_df], axis=1)

    logger.info("Polynomial features added (degree %d); columns now: %d", degree, len(df.columns))
    return df
