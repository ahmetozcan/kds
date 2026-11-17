"""Fold-level and record-level significance analysis for the KDS ablation.

Scores the three configurations of the ablation on the shared fold set of the
reported run and reports the paired contrasts of Table ``tab:sig``. Nothing is
refitted: the stored out-of-fold predictions are read from disk.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

RNG = np.random.default_rng(42)

def folds(n):
    return [v for _, v in KFold(5, shuffle=True, random_state=42).split(np.arange(n))]

def load():
    fk  = pd.read_csv("data/processed/sig_a_off/features_kalman.csv")
    m_off = pd.read_csv("data/processed/sig_a_off/meta_features.csv")
    m_on  = pd.read_csv("data/processed/sig_a/meta_features.csv")
    y = fk["rop_mm_m"].to_numpy()
    assert np.allclose(y, m_off["rop_mm_m"]) and np.allclose(y, m_on["rop_mm_m"])
    return y, {
        "C1 single learner":     fk["rop_kf_tabpfn"].to_numpy(),
        "C2 unfiltered ensemble": m_off["tabpfn"].to_numpy(),
        "C3 filtered ensemble":   m_on["tabpfn"].to_numpy(),
    }

def holm(pvals):
    order = np.argsort(pvals); adj = np.empty(len(pvals)); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj

def block_bootstrap_dmae(e_a, e_b, block, n_boot=10000):
    """Moving-block bootstrap CI for MAE(a) - MAE(b) on paired absolute errors."""
    d = e_a - e_b; n = len(d); nblocks = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(n_boot, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]), means.std(ddof=1)

y, preds = load()
fl = folds(len(y))
names = list(preds)
print("=" * 78)
print(f"SHARED FOLD SET  (shuffled 5-fold, n={len(y)} records, k=5)")
print("=" * 78)

per_fold = {k: {"r2": [], "mae": []} for k in names}
for v in fl:
    for k in names:
        per_fold[k]["r2"].append(r2_score(y[v], preds[k][v]))
        per_fold[k]["mae"].append(mean_absolute_error(y[v], preds[k][v]))
print("\nPer-fold R^2")
print(f"{'configuration':24s}" + "".join(f"{'f'+str(i+1):>9s}" for i in range(5)) + f"{'mean':>10s}{'sd':>8s}")
for k in names:
    r = np.array(per_fold[k]["r2"])
    print(f"{k:24s}" + "".join(f"{x:9.4f}" for x in r) + f"{r.mean():10.4f}{r.std(ddof=1):8.4f}")
print("\nPer-fold MAE (mm/min)")
for k in names:
    m = np.array(per_fold[k]["mae"])
    print(f"{k:24s}" + "".join(f"{x:9.4f}" for x in m) + f"{m.mean():10.4f}{m.std(ddof=1):8.4f}")

contrasts = [("C3 filtered ensemble", "C2 unfiltered ensemble"),
             ("C3 filtered ensemble", "C1 single learner"),
             ("C2 unfiltered ensemble", "C1 single learner")]

print("\n--- Fold-level paired t-test on R^2 (n=5 folds) ---")
raw = []
for hi, lo in contrasts:
    d = np.array(per_fold[hi]["r2"]) - np.array(per_fold[lo]["r2"])
    t, p = stats.ttest_rel(per_fold[hi]["r2"], per_fold[lo]["r2"])
    dz = d.mean() / d.std(ddof=1)
    ci = stats.t.interval(0.95, 4, loc=d.mean(), scale=d.std(ddof=1)/np.sqrt(5))
    w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    raw.append(p)
    print(f"  {hi.split()[0]} vs {lo.split()[0]}: mean_d={d.mean():+.4f} "
          f"95%CI[{ci[0]:+.4f},{ci[1]:+.4f}] t={t:+.3f} p={p:.4f} d_z={dz:+.2f} "
          f"| wins {int((d>0).sum())}/5 | Wilcoxon p={w.pvalue:.4f}")
for (hi, lo), p in zip(contrasts, holm(np.array(raw))):
    print(f"  Holm-adjusted p  {hi.split()[0]} vs {lo.split()[0]}: {p:.4f}")

print("\n--- Record-level paired absolute errors (n=%d) ---" % len(y))
errs = {k: np.abs(y - preds[k]) for k in names}
raw2 = []
for hi, lo in contrasts:
    ea, eb = errs[lo], errs[hi]           # positive => hi has lower error
    d = ea - eb
    w = stats.wilcoxon(d, alternative="two-sided")
    raw2.append(w.pvalue)
    ci50, sd50 = block_bootstrap_dmae(ea, eb, 50)
    ci20, _ = block_bootstrap_dmae(ea, eb, 20)
    ci100, _ = block_bootstrap_dmae(ea, eb, 100)
    print(f"  {hi.split()[0]} vs {lo.split()[0]}: dMAE={d.mean():+.4f} mm/min "
          f"Wilcoxon p={w.pvalue:.3g}")
    print(f"      block-bootstrap 95%CI  b=20 [{ci20[0]:+.4f},{ci20[1]:+.4f}]  "
          f"b=50 [{ci50[0]:+.4f},{ci50[1]:+.4f}]  b=100 [{ci100[0]:+.4f},{ci100[1]:+.4f}]")
for (hi, lo), p in zip(contrasts, holm(np.array(raw2))):
    print(f"  Holm-adjusted p  {hi.split()[0]} vs {lo.split()[0]}: {p:.3g}")
print()
