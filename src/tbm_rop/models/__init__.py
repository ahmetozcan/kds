"""Model registry and factory.

Each entry is a constructor ``(params: dict) -> estimator`` returning a
scikit-learn-compatible regressor (``fit``/``predict``). Hyperparameters come
from config or tuning output; constructors only translate them into estimator
instances.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

ModelConstructor = Callable[[dict[str, Any]], Any]

MODEL_REGISTRY: dict[str, ModelConstructor] = {}


def register_model(name: str) -> Callable[[ModelConstructor], ModelConstructor]:
    """Register a model constructor under ``name``."""

    def decorator(func: ModelConstructor) -> ModelConstructor:
        MODEL_REGISTRY[name] = func
        return func

    return decorator


def create_model(name: str, params: dict[str, Any] | None = None) -> Any:
    """Instantiate a registered model.

    Args:
        name: Registered model name (e.g. ``"random_forest"``).
        params: Hyperparameters passed to the constructor.

    Returns:
        A fitted-API estimator instance.

    Raises:
        KeyError: When the model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](params or {})


from . import mlp, tabpfn_model, tree  # noqa: E402,F401  (register models)

__all__ = ["MODEL_REGISTRY", "register_model", "create_model"]
