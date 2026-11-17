"""TabPFN foundation-model constructor."""

import logging
from typing import Any

import torch

from . import register_model

logger = logging.getLogger(__name__)


@register_model("tabpfn")
def build_tabpfn(params: dict[str, Any]) -> Any:
    """TabPFN regressor; device defaults to CUDA when available.

    Raises:
        ImportError: When the ``tabpfn`` package is not installed.
    """
    from tabpfn import TabPFNRegressor

    defaults: dict[str, Any] = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "ignore_pretraining_limits": True,
    }
    merged = {**defaults, **params}
    logger.info("TabPFN on device: %s", merged["device"])
    return TabPFNRegressor(**merged)
