"""Re-filter raw OOF prediction columns with a given Kalman process noise Q.

Part of the Q sensitivity sweep (plan/tuning-plan.md, Task 4c). The raw OOF
series are generated once (kalman.mode=none => rop_kf_* columns hold raw OOF);
this script applies the causal filter with a chosen Q and writes a new
features_kalman.csv, after which only the cheap `stack` stage needs to run.

Usage:
    uv run python run/pipeline/q_filter.py \
        --source data/processed/qs_raw/features_kalman.csv \
        --q 0.1 --out data/processed/qs_0.1
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from tbm_rop.cv import CVConfig, make_folds
from tbm_rop.data import TARGET_COLUMN
from tbm_rop.kalman import denoise_oof_foldwise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("q_filter")

OOF_PREFIX = "rop_kf_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="features_kalman.csv with RAW OOF columns")
    parser.add_argument("--q", type=float, required=True, help="Kalman process noise Q")
    parser.add_argument("--out", required=True, help="Output processed dir")
    parser.add_argument("--scheme", default="shuffled", help="CV scheme for fold-local R")
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds")
    parser.add_argument("--random-state", type=int, default=42, help="CV seed")
    parser.add_argument("--gap", type=int, default=0, help="Purge gap (blocked scheme)")
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    y = df[TARGET_COLUMN]
    oof_cols = [c for c in df.columns if c.startswith(OOF_PREFIX)]
    if not oof_cols:
        raise ValueError(f"No {OOF_PREFIX}* columns found in {args.source}")

    folds = make_folds(
        len(df),
        CVConfig(
            scheme=args.scheme,
            n_splits=args.n_splits,
            random_state=args.random_state,
            gap=args.gap,
        ),
    )

    for col in oof_cols:
        # R is estimated per fold on training rows only (see denoise stage).
        df[col], r_per_fold = denoise_oof_foldwise(
            df[col].to_numpy(),
            y.to_numpy(dtype=float),
            folds,
            process_noise=args.q,
            mode="causal",
        )
        logger.info(
            "Filtered %s with Q=%g (fold-local R: %.4f..%.4f)",
            col, args.q, min(r_per_fold), max(r_per_fold),
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_kalman.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows, %d cols)", out_path, *df.shape)


if __name__ == "__main__":
    main()
