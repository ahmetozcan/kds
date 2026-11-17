"""Significance analysis for the nested axis of the KDS ablation.

Mirrors ``significance_tests.py`` but takes the two ensemble configurations
from the nested runs instead of the shared-fold ones. Nothing is refitted:
the per-record predictions written by ``stage_nested`` are read from disk.

The single-learner control (C1) is unchanged, because a standalone base
learner never touches the augmented table and is therefore identical on
both axes.

Two integrator choices are reported, since they answer different questions:
  ``fixed``    -- TabPFN integrates in both C2 and C3, exactly as in
                  ``significance_tests.py``, so the contrast isolates the
                  filter and the result is comparable to Section ``sec:sig``;
  ``champion`` -- each cell uses its own tournament winner, matching the
                  figures quoted in Table ``tab:nested``, so this contrast
                  mixes the filter with the integrator choice.

On the reported run TabPFN wins both cells, so the two blocks print the same
figures. The distinction is kept because the champion is read from the report
rather than assumed.

Usage:
    .venv/bin/python run/pipeline/significance_nested.py
"""

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

RNG = np.random.default_rng(42)
TAG = "a"  # the reported run; its nested predictions are in data/processed/a


def folds(n: int) -> list[np.ndarray]:
    return [v for _, v in KFold(5, shuffle=True, random_state=42).split(np.arange(n))]


def _champion(run: str) -> str:
    with open(f"outputs/nested_{run}/nested_report.json") as fh:
        return json.load(fh)["champion"]["name"]


def load(tag: str, integrator: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    """Return targets, the three configurations, and which column each used."""
    off = f"{tag}_off"
    single = pd.read_csv(f"data/processed/sig_{off}/features_kalman.csv")
    on_df = pd.read_csv(f"data/processed/{tag}/nested_predictions.csv")
    off_df = pd.read_csv(f"data/processed/{off}/nested_predictions.csv")

    y = single["rop_mm_m"].to_numpy()
    for name, df in (("nested filtered", on_df), ("nested unfiltered", off_df)):
        if not np.allclose(y, df["rop_mm_m"].to_numpy()):
            raise ValueError(f"{name} targets are not aligned with the control run")

    if integrator == "fixed":
        col_on = col_off = "tabpfn"
    else:
        col_on, col_off = _champion(tag), _champion(off)

    preds = {
        "C1 single learner": single["rop_kf_tabpfn"].to_numpy(),
        "C2 unfiltered ensemble": off_df[col_off].to_numpy(),
        "C3 filtered ensemble": on_df[col_on].to_numpy(),
    }
    used = {"C2 unfiltered ensemble": col_off, "C3 filtered ensemble": col_on,
            "C1 single learner": "tabpfn (base learner)"}
    return y, preds, used


def holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def block_bootstrap_dmae(e_a: np.ndarray, e_b: np.ndarray, block: int,
                         n_boot: int = 10000) -> tuple[np.ndarray, float]:
    """Moving-block bootstrap CI for MAE(a) - MAE(b) on paired absolute errors."""
    d = e_a - e_b
    n = len(d)
    nblocks = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(n_boot, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]), means.std(ddof=1)


CONTRASTS = [("C3 filtered ensemble", "C2 unfiltered ensemble"),
             ("C3 filtered ensemble", "C1 single learner"),
             ("C2 unfiltered ensemble", "C1 single learner")]


def report(tag: str, integrator: str) -> None:
    y, preds, used = load(tag, integrator)
    fl = folds(len(y))
    names = list(preds)

    print("=" * 78)
    print(f"NESTED  integrator={integrator}  "
          f"(shuffled 5-fold, n={len(y)} records, k=5)")
    for k in names:
        print(f"    {k:24s} <- {used[k]}")
    print("=" * 78)

    per_fold = {k: {"r2": [], "mae": []} for k in names}
    for v in fl:
        for k in names:
            per_fold[k]["r2"].append(r2_score(y[v], preds[k][v]))
            per_fold[k]["mae"].append(mean_absolute_error(y[v], preds[k][v]))

    print("\nPer-fold R^2")
    print(f"{'configuration':24s}" + "".join(f"{'f' + str(i + 1):>9s}" for i in range(5))
          + f"{'mean':>10s}{'sd':>8s}")
    for k in names:
        r = np.array(per_fold[k]["r2"])
        print(f"{k:24s}" + "".join(f"{x:9.4f}" for x in r)
              + f"{r.mean():10.4f}{r.std(ddof=1):8.4f}")
    print("\nPer-fold MAE (mm/min)")
    for k in names:
        m = np.array(per_fold[k]["mae"])
        print(f"{k:24s}" + "".join(f"{x:9.4f}" for x in m)
              + f"{m.mean():10.4f}{m.std(ddof=1):8.4f}")

    print("\n--- Fold-level paired t-test on R^2 (n=5 folds) ---")
    raw = []
    for hi, lo in CONTRASTS:
        d = np.array(per_fold[hi]["r2"]) - np.array(per_fold[lo]["r2"])
        t, p = stats.ttest_rel(per_fold[hi]["r2"], per_fold[lo]["r2"])
        dz = d.mean() / d.std(ddof=1)
        ci = stats.t.interval(0.95, 4, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(5))
        w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        raw.append(p)
        print(f"  {hi.split()[0]} vs {lo.split()[0]}: mean_d={d.mean():+.4f} "
              f"95%CI[{ci[0]:+.4f},{ci[1]:+.4f}] t={t:+.3f} p={p:.4f} d_z={dz:+.2f} "
              f"| wins {int((d > 0).sum())}/5 | Wilcoxon p={w.pvalue:.4f}")
    for (hi, lo), p in zip(CONTRASTS, holm(np.array(raw))):
        print(f"  Holm-adjusted p  {hi.split()[0]} vs {lo.split()[0]}: {p:.4f}")

    print(f"\n--- Record-level paired absolute errors (n={len(y)}) ---")
    errs = {k: np.abs(y - preds[k]) for k in names}
    raw2 = []
    for hi, lo in CONTRASTS:
        ea, eb = errs[lo], errs[hi]           # positive => hi has lower error
        d = ea - eb
        w = stats.wilcoxon(d, alternative="two-sided")
        raw2.append(w.pvalue)
        ci50, _ = block_bootstrap_dmae(ea, eb, 50)
        ci20, _ = block_bootstrap_dmae(ea, eb, 20)
        ci100, _ = block_bootstrap_dmae(ea, eb, 100)
        print(f"  {hi.split()[0]} vs {lo.split()[0]}: dMAE={d.mean():+.4f} mm/min "
              f"Wilcoxon p={w.pvalue:.3g}")
        print(f"      block-bootstrap 95%CI  b=20 [{ci20[0]:+.4f},{ci20[1]:+.4f}]  "
              f"b=50 [{ci50[0]:+.4f},{ci50[1]:+.4f}]  b=100 [{ci100[0]:+.4f},{ci100[1]:+.4f}]")
    for (hi, lo), p in zip(CONTRASTS, holm(np.array(raw2))):
        print(f"  Holm-adjusted p  {hi.split()[0]} vs {lo.split()[0]}: {p:.3g}")
    print()


def main() -> None:
    for integrator in ("fixed", "champion"):
        report(TAG, integrator)


if __name__ == "__main__":
    main()
