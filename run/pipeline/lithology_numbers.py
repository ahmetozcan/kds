"""Regenerate the lithological table of the supplementary sheet.

The released file names no lithological unit, but UCS takes exactly eight
distinct values, one per unit of the site investigation, and each occupies
contiguous runs of records. Keying on UCS therefore recovers the units, and
this script rebuilds every column of Table ``tab:lithology`` from the feature
table: the abrasivity index, the number of separate zones in which a unit is
met, how many records it holds, and the mean and spread of its ROP. The MAE of
the reported run is printed beside them, which the paper does not tabulate.

Usage:
    .venv/bin/python run/pipeline/lithology_numbers.py
"""

import numpy as np
import pandas as pd

LITHOLOGY = {191.4: "Quartzite", 122.6: "Slate", 101.9: "Slate-metasandstone",
             101.0: "Quartzite-schist", 81.2: "Metasandstone",
             71.5: "Slate-metasandstone-schist", 46.0: "Metasandstone-schist",
             10.7: "Schist"}


def _load() -> tuple[pd.DataFrame, np.ndarray]:
    fk = pd.read_csv("data/processed/sig_a/features_kalman.csv")
    pred = pd.read_csv("data/processed/sig_a/meta_features.csv")["tabpfn"].to_numpy()
    return fk, pred


def _zones(mask: np.ndarray) -> int:
    """Number of contiguous runs of records in which a unit is met."""
    return int(np.count_nonzero(np.diff(np.r_[0, mask.astype(int), 0]) == 1))


def main() -> None:
    fk, pred = _load()
    y = fk["rop_mm_m"].to_numpy()
    ucs = fk["ucs_mpa"].round(1).to_numpy()
    cai = fk["cai"].to_numpy()

    print("=== tab:lithology ===")
    print(f"{'Unit':28s}{'UCS':>7s}{'CAI':>6s}{'Zones':>7s}{'Rec.':>7s}"
          f"{'Mean':>8s}{'Std':>7s}{'MAE':>7s}")
    for u, name in LITHOLOGY.items():
        m = ucs == u
        print(f"{name:28s}{u:7.1f}{cai[m][0]:6.2f}{_zones(m):7d}{int(m.sum()):7d}"
              f"{y[m].mean():8.1f}{y[m].std(ddof=1):7.1f}"
              f"{np.abs(y[m] - pred[m]).mean():7.2f}")
    print(f"{'All records':28s}{'':7s}{'':6s}{'':7s}{len(y):7d}"
          f"{y.mean():8.1f}{y.std(ddof=1):7.1f}{np.abs(y - pred).mean():7.2f}")


if __name__ == "__main__":
    main()
