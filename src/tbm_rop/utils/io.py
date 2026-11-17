"""Artifact I/O: intermediate outputs are saved with sidecar metadata."""

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def file_sha256(path: str | Path, length: int = 12) -> str:
    """Short SHA256 of a file, for dataset/artifact fingerprinting."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()[:length]


def save_artifact(df: pd.DataFrame, path: str | Path, meta: dict | None = None) -> Path:
    """Save a DataFrame as CSV plus a ``.meta.json`` sidecar.

    The sidecar records shape, column list, content hash, and any
    caller-provided metadata (config, source files, stage name).

    Args:
        df: Frame to persist (index is not written).
        path: Target CSV path; parents are created.
        meta: Extra metadata merged into the sidecar.

    Returns:
        The resolved CSV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    sidecar = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": len(df),
        "columns": df.columns.tolist(),
        "sha256": file_sha256(path),
        **(meta or {}),
    }
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    logger.info("Artifact saved: %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return path
