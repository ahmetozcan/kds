"""Regenerate every accuracy figure quoted in the paper from the reports.

The manuscript quotes roughly forty numbers that all trace back to a
``stack_report.json``, a ``denoise_report.json``, a ``nested_report.json`` or
the denoised feature tables. Recomputing them in one place makes a numeric
update pass mechanical and keeps provenance explicit: each line names the
artifact it came from.

Usage:
    python run/pipeline/paper_numbers.py [--root outputs] [--processed data/processed]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("paper_numbers")

Q_VALUES = ["0.01", "0.03", "0.1", "0.3", "1.0", "3.0"]


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _champ(root: Path, run: str, report: str = "stack_report.json") -> tuple[str, float] | None:
    data = _load(root / run / report)
    if not data or not data.get("champion"):
        return None
    return str(data["champion"]["name"]), float(data["champion"]["r2"])


def _champ_metrics(root: Path, run: str) -> dict[str, float] | None:
    """MAE and RMSE of the champion configuration of a stack report."""
    data = _load(root / run / "stack_report.json")
    if not data or not data.get("champion"):
        return None
    name = data["champion"]["name"]
    individual = data.get("individual_base_models") or {}
    if name in individual:
        return {k: float(v) for k, v in individual[name].items()}
    return None


def _raw_oof(root: Path, run: str, model: str) -> dict[str, float] | None:
    """Unaugmented out-of-fold metrics of one base learner."""
    data = _load(root / run / "denoise_report.json")
    if not data:
        return None
    entry = (data.get("scores") or {}).get(model)
    return {k: float(v) for k, v in entry["raw_oof"].items()} if entry else None


def _nested_tuned(root: Path, run: str, model: str) -> float | None:
    """Nested-tuning OOF R2 of one learner (source of the per-integrator table)."""
    data = _load(root / run / "tuning_report.json")
    if not data:
        return None
    entry = (data.get("models") or {}).get(model)
    return float(entry["nested_oof"]["r2"]) if entry else None


def _smoothing_stat(
    processed: Path, on: str, off: str, column: str
) -> tuple[float, float, float] | None:
    """Mean absolute record-to-record change, unfiltered vs filtered."""
    try:
        raw = pd.read_csv(processed / off / "features_kalman.csv")[column].to_numpy()
        filt = pd.read_csv(processed / on / "features_kalman.csv")[column].to_numpy()
    except (OSError, KeyError):
        return None
    before = float(np.abs(np.diff(raw)).mean())
    after = float(np.abs(np.diff(filt)).mean())
    return before, after, 100.0 * (1.0 - after / before)


def _line(label: str, value: tuple[str, float] | None, source: str) -> None:
    if value is None:
        print(f"  {label:34s} {'(absent)':>12s}   {source}")
    else:
        print(f"  {label:34s} {value[1]:12.6f}   {source}  [{value[0]}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--processed", default="data/processed")
    args = parser.parse_args()
    root, processed = Path(args.root), Path(args.processed)

    print("=== Tournament winners ===")
    for run, label in [
        ("sig_a", "Reported run, Q=0.1, filter on"),
        ("sig_a_off", "Same pipeline, filter off"),
        ("a", "Tuning-stage run, filter on"),
        ("a_off", "Tuning-stage run, filter off"),
    ]:
        _line(label, _champ(root, run), f"{run}/stack_report.json")

    print("\n=== Filter on against filter off (the ablation gain) ===")
    for on, off in [("sig_a", "sig_a_off"), ("a", "a_off")]:
        c_on, c_off = _champ(root, on), _champ(root, off)
        if c_on and c_off:
            print(
                f"  {on:12s} filtered {c_on[1]:.4f}  unfiltered {c_off[1]:.4f}"
                f"  gain {c_on[1] - c_off[1]:+.4f}"
            )

    print("\n=== Process-noise sweep ===")
    for q in Q_VALUES:
        _line(f"Q = {q}", _champ(root, f"qs_{q}"), f"qs_{q}/stack_report.json")
    best_meta = _load(root / "qs_0.03" / "stack_report.json")
    if best_meta and best_meta.get("meta_tournament_r2"):
        name, r2 = max(best_meta["meta_tournament_r2"].items(), key=lambda kv: kv[1])
        print(f"  {'Q=0.03, best integrator':34s} {r2:12.6f}   qs_0.03  [{name}]")

    print("\n=== Single learner before the architecture (raw OOF) ===")
    metrics = _raw_oof(root, "sig_a_off", "tabpfn")
    if metrics:
        print(
            f"  {'TabPFN raw OOF':34s} {metrics['r2']:12.6f}   sig_a_off/denoise_report.json"
            f"  (MAE {metrics['mae']:.3f}, RMSE {metrics['rmse']:.3f})"
        )

    print("\n=== Winner MAE and RMSE (ablation table) ===")
    for run in ["sig_a", "sig_a_off"]:
        metrics = _champ_metrics(root, run)
        if metrics:
            print(
                f"  {run:34s} MAE {metrics['mae']:7.3f}  RMSE {metrics['rmse']:7.3f}"
                f"   {run}/stack_report.json"
            )

    print("\n=== Nested construction of the prediction columns ===")
    for run in ["nested_a", "nested_a_off"]:
        _line(run, _champ(root, run, "nested_report.json"), f"{run}/nested_report.json")

    print("\n=== Filtered against unfiltered, per integrator ===")
    print(f"  {'model':15s} {'unfiltered':>12s} {'filtered':>11s} {'diff':>9s}")

    def _pair(unfiltered: float | None, filtered: float | None) -> str:
        """Format the two cells of one learner and their difference."""
        if unfiltered is None or filtered is None:
            return f"{'--':>12s} {'--':>11s} {'--':>9s}"
        return f"{unfiltered:12.4f} {filtered:11.4f} {filtered - unfiltered:+9.4f}"

    c_on, c_off = _champ(root, "a"), _champ(root, "a_off")
    if c_on or c_off:
        print(
            f"  {'TabPFN*':15s} "
            f"{_pair(c_off[1] if c_off else None, c_on[1] if c_on else None)}"
            f"   (stack_report)"
        )
    for model in ("catboost", "lightgbm", "random_forest", "mlp"):
        cells = [_nested_tuned(root, r, model) for r in ("a_off", "a")]
        if all(c is None for c in cells):
            print(f"  {model:15s} (no tuning_report in this artifact set)")
            continue
        print(f"  {model:15s} {_pair(cells[0], cells[1])}   (tuning_report)")

    print("\n=== Smoothing effect of the filter ===")
    stat = _smoothing_stat(processed, "sig_a", "sig_a_off", "rop_kf_tabpfn")
    if stat:
        before, after, pct = stat
        print(
            f"  mean absolute record-to-record change  {before:.4f} -> {after:.4f} mm/min"
            f"  ({pct:.1f}% reduction)   sig_a / sig_a_off"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
