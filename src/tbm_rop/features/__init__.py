"""Feature-builder registry.

Each builder is a function ``(df, params) -> df`` registered under a name.
``build_features`` applies a configured sequence of builders in order.
"""

import logging
from collections.abc import Callable

import pandas as pd

logger = logging.getLogger(__name__)

FeatureBuilder = Callable[[pd.DataFrame, dict | None], pd.DataFrame]

FEATURE_BUILDER_REGISTRY: dict[str, FeatureBuilder] = {}


def register_feature_builder(name: str) -> Callable[[FeatureBuilder], FeatureBuilder]:
    """Register a feature builder under ``name``."""

    def decorator(func: FeatureBuilder) -> FeatureBuilder:
        FEATURE_BUILDER_REGISTRY[name] = func
        return func

    return decorator


def build_features(
    df: pd.DataFrame,
    steps: list[str] | None = None,
    step_params: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Apply feature builders in sequence.

    Args:
        df: Input frame.
        steps: Builder names in application order. Defaults to the full
            original pipeline: physics -> temporal -> clustering -> polynomial.
        step_params: Optional per-step parameter dicts keyed by builder name.

    Returns:
        Feature-enriched frame.

    Raises:
        KeyError: When an unknown builder name is requested.
    """
    steps = steps or ["physics", "temporal", "clustering", "polynomial"]
    step_params = step_params or {}

    for step in steps:
        if step not in FEATURE_BUILDER_REGISTRY:
            raise KeyError(
                f"Unknown feature builder '{step}'. Available: {sorted(FEATURE_BUILDER_REGISTRY)}"
            )
        df = FEATURE_BUILDER_REGISTRY[step](df, step_params.get(step))
        logger.info("Applied feature step '%s'", step)
    return df


from . import clustering, physics, polynomial, temporal  # noqa: E402,F401  (register builders)

__all__ = [
    "FEATURE_BUILDER_REGISTRY",
    "register_feature_builder",
    "build_features",
]
