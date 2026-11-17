"""Cross-validation schemes for chainage-ordered tunnel data.

Two schemes are provided:

- ``shuffled``: standard shuffled KFold. Comparable to prior work on this
  dataset, but optimistic for serially correlated data because neighbouring
  rows land in different folds.
- ``blocked``: contiguous chainage blocks with an optional purge ``gap``
  around each validation block. Robust to spatial autocorrelation leakage.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

Fold = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CVConfig:
    """Cross-validation configuration.

    Attributes:
        scheme: ``"shuffled"`` or ``"blocked"``.
        n_splits: Number of folds.
        random_state: Seed for the shuffled scheme.
        gap: Purge width (rows) removed from training on each side of a
            validation block. Only used by the blocked scheme.
    """

    scheme: str = "shuffled"
    n_splits: int = 5
    random_state: int = 42
    gap: int = 0


def make_folds(n_samples: int, cfg: CVConfig) -> list[Fold]:
    """Build train/validation index pairs for the configured scheme.

    Args:
        n_samples: Number of rows in the dataset (chainage order assumed).
        cfg: Cross-validation configuration.

    Returns:
        List of ``(train_idx, val_idx)`` tuples.

    Raises:
        ValueError: For an unknown scheme.
    """
    if cfg.scheme == "shuffled":
        kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        return [(train, val) for train, val in kf.split(np.arange(n_samples))]

    if cfg.scheme == "blocked":
        return _blocked_folds(n_samples, cfg.n_splits, cfg.gap)

    raise ValueError(f"Unknown CV scheme: {cfg.scheme!r} (use 'shuffled' or 'blocked').")


def _blocked_folds(n_samples: int, n_splits: int, gap: int) -> list[Fold]:
    """Contiguous validation blocks with a purge gap around each block."""
    indices = np.arange(n_samples)
    blocks = np.array_split(indices, n_splits)

    folds: list[Fold] = []
    for block in blocks:
        val_start, val_end = block[0], block[-1]
        purge_start, purge_end = val_start - gap, val_end + gap
        train_mask = (indices < purge_start) | (indices > purge_end)
        folds.append((indices[train_mask], block))
        logger.debug(
            "Blocked fold: val [%d, %d], train size %d", val_start, val_end, train_mask.sum()
        )
    return folds
