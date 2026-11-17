"""Operating-regime clustering feature (KMeans cluster membership)."""

import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import register_feature_builder

logger = logging.getLogger(__name__)

DEFAULT_COLUMNS = ["rpm_rotate_per_minute", "chd_knm", "thrust_kn", "ucs_mpa", "cai", "weathering"]
DEFAULT_N_CLUSTERS = 5


@register_feature_builder("clustering")
def add_cluster_feature(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Add a ``cluster_id`` column from KMeans on standardized input variables.

    Args:
        df: Input frame.
        params: Optional overrides: ``columns``, ``n_clusters``, ``random_state``.

    Returns:
        Copy of ``df`` with a ``cluster_id`` column. Rows with NaNs in the
        clustering columns are dropped (mirrors the original pipeline).
    """
    params = params or {}
    columns = params.get("columns", DEFAULT_COLUMNS)
    n_clusters = params.get("n_clusters", DEFAULT_N_CLUSTERS)
    random_state = params.get("random_state", 42)

    df = df.copy()
    df = df.dropna(subset=columns)

    scaled = StandardScaler().fit_transform(df[columns])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(scaled)

    logger.info("Cluster feature added (%d clusters); rows: %d", n_clusters, len(df))
    return df
