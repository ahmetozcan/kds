"""Random-seed management for reproducibility."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs and force deterministic cuDNN.

    Deterministic cuDNN kernels make GPU training bit-reproducible on the
    same hardware/driver at a small speed cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d (deterministic cuDNN enabled)", seed)
