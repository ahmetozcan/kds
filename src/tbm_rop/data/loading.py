"""Raw dataset loading and column standardization."""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "rop_mm_m"

# Columns expected in the raw ENGEO/Gokceoglu 2021 dataset after cleaning.
EXPECTED_COLUMNS = [
    "rop_mm_m",
    "rpm_rotate_per_minute",
    "chd_knm",
    "thrust_kn",
    "alfa_degree",
    "watinflow_l_m",
    "weathering",
    "cai",
    "ucs_mpa",
    "ucs_weat",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, e.g. 'CHD (kNm)' -> 'chd_knm'."""
    mapping = {}
    for col in df.columns:
        new_col = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
        mapping[col] = new_col
    return df.rename(columns=mapping)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw Excel dataset, clean names, and validate the schema.

    Args:
        path: Path to the raw .xlsx file.

    Returns:
        DataFrame with standardized column names, original row order preserved.

    Raises:
        FileNotFoundError: When the raw file does not exist.
        ValueError: When expected columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = pd.read_excel(path)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df = clean_column_names(df)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing expected columns: {missing}")

    logger.info("Loaded raw dataset: %d rows, %d columns", len(df), len(df.columns))
    return df.reset_index(drop=True)
