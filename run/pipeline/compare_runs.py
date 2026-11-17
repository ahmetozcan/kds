"""Compare post-leakage-fix results against the archived pre-fix reports.

Every number quoted in the paper comes from a ``stack_report.json`` or a
``nested_report.json``. This script pairs each current report with its
archived counterpart and prints the deltas, so the claim that the fixes leave
the conclusions intact is checked rather than asserted.

Usage:
    python run/pipeline/compare_runs.py [--archive DIR] [--max-delta 0.01]

Exit status is 1 when any champion moves by more than ``--max-delta``, which
makes the script usable as a guard in a re-run driver.
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare")


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _champion(report: dict) -> tuple[str, float] | None:
    champ = report.get("champion")
    if not champ:
        return None
    return str(champ.get("name")), float(champ.get("r2"))


def _scores(report: dict) -> dict[str, float]:
    """Flatten every scored configuration in a report into name -> R²."""
    out: dict[str, float] = {}
    for name, metrics in (report.get("individual_base_models") or {}).items():
        out[name] = float(metrics["r2"])
    for name, r2 in (report.get("meta_tournament_r2") or {}).items():
        out[f"stacking_meta_{name}"] = float(r2)
    for name, metrics in (report.get("level2_nested") or {}).items():
        out[f"nested_{name}"] = float(metrics["r2"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs", help="Current results root")
    parser.add_argument(
        "--archive", default="archive/pre-leakfix-2026-10-25", help="Archived results root"
    )
    parser.add_argument(
        "--max-delta", type=float, default=0.01, help="Champion drift that fails the check"
    )
    args = parser.parse_args()

    outputs, archive = Path(args.outputs), Path(args.archive)
    worst_name, worst_delta = "", 0.0
    unmatched: list[str] = []

    for report_name in ("stack_report.json", "nested_report.json"):
        for current_path in sorted(outputs.glob(f"*/{report_name}")):
            run = current_path.parent.name
            current = _load(current_path)
            if current is None:
                continue
            archived = _load(archive / run / report_name)
            if archived is None:
                champ = _champion(current)
                label = f"{champ[0]} {champ[1]:.6f}" if champ else "-"
                unmatched.append(f"{run}/{report_name}: {label} (arsivde karsiligi yok)")
                continue

            print(f"\n=== {run} ({report_name}) ===")
            before, after = _scores(archived), _scores(current)
            for key in sorted(after, key=lambda k: -after[k]):
                if key in before:
                    delta = after[key] - before[key]
                    print(f"  {key:26s} {before[key]:10.6f} -> {after[key]:10.6f}  {delta:+.6f}")
                else:
                    print(f"  {key:26s} {'-':>10s} -> {after[key]:10.6f}  (yeni)")

            champ_before, champ_after = _champion(archived), _champion(current)
            if champ_before and champ_after:
                delta = champ_after[1] - champ_before[1]
                flag = "  <-- ETIKET DEGISTI" if champ_before[0] != champ_after[0] else ""
                print(
                    f"  {'SAMPIYON':26s} {champ_before[1]:10.6f} -> {champ_after[1]:10.6f} "
                    f" {delta:+.6f}{flag}"
                )
                print(f"  {'':26s} {champ_before[0]:>10s} -> {champ_after[0]:>10s}")
                if abs(delta) > abs(worst_delta):
                    worst_name, worst_delta = run, delta

    if unmatched:
        print("\n=== arsivde esi olmayan raporlar ===")
        for line in unmatched:
            print(f"  {line}")

    print(f"\nEn buyuk sampiyon kaymasi: {worst_name} {worst_delta:+.6f}")
    if abs(worst_delta) > args.max_delta:
        print(f"UYARI: {args.max_delta} esigini asiyor.")
        return 1
    print(f"Tum sampiyonlar {args.max_delta} esiginin altinda.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
