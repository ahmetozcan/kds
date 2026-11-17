"""Write the cross-validation assignment of every record to a CSV.

The folds are deterministic given the seed, so this file is a convenience
rather than a dependency: it lets a reader check a split without running
Python, and lets an external re-implementation align with the one used in
the paper.

Usage:
    uv run python run/pipeline/export_folds.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tbm_rop.cv import CVConfig, make_folds

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("export_folds")

FEATURES = Path("data/processed/sig_a/features_kalman.csv")
OUT = Path("artifacts/folds.csv")


def assignment(n: int) -> np.ndarray:
    """Return the validation fold index of each record."""
    folds = make_folds(n, CVConfig(scheme="shuffled", n_splits=5, random_state=42))
    out = np.full(n, -1, dtype=int)
    for k, (_, val) in enumerate(folds):
        out[val] = k
    if (out < 0).any():
        raise ValueError(f"{(out < 0).sum()} records left unassigned")
    return out


def main() -> None:
    n = len(pd.read_csv(FEATURES, usecols=["rop_mm_m"]))
    frame = pd.DataFrame(
        {
            "record": np.arange(n),
            "fold": assignment(n),
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    sizes = frame["fold"].value_counts().sort_index().to_list()
    logger.info("wrote %s for %d records; fold sizes %s", OUT, n, sizes)


if __name__ == "__main__":
    main()
