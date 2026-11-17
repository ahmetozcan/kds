"""Runtime environment capture for reproducibility reports."""

import logging
import platform
from importlib import metadata

import torch

logger = logging.getLogger(__name__)

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "catboost",
    "tabpfn",
    "optuna",
    "torch",
)


def collect_environment() -> dict:
    """Snapshot interpreter, hardware, and key package versions.

    Returns:
        JSON-serializable dict embedded into every experiment report so that
        results can be traced back to the exact software/hardware stack.
    """
    packages: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"

    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "gpu_model": torch.cuda.get_device_name(0) if cuda_available else "N/A",
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "packages": packages,
    }
