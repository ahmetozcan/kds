"""Generate the manuscript figures from stored out-of-fold predictions.

Produces:
  fig_windows.pdf     - the ensemble with and without filtering, at four
                        lithological boundaries
  fig_denoising.pdf   - superseded by fig_windows; still used by the frozen
                        Turkish copy of the manuscript
  fig_longsection_full.pdf - both level-1 signals and the level-2 output over
                        the whole section, with the lithological section
                        (standalone A3 supplementary sheet)
  fig_accuracy.pdf    - measured vs predicted, and the error distribution
  fig_qsweep.pdf      - sensitivity to the process noise Q

Every figure of the manuscript is redrawn here. fig_denoising is the one
exception to that correspondence: it was superseded by fig_windows and is kept
only because the frozen Turkish copy still carries it.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Okabe-Ito colourblind-safe palette
GREY, ORANGE, BLUE, GREEN = "#999999", "#E69F00", "#0072B2", "#009E73"
OUT = Path("paper/figures")
SPACING_M = 1.5  # records are 1.5 m of advance apart (Gokceoglu 2021);
# the dataset carries no chainage column, so distance is measured from the
# first record rather than from the project datum.

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def load():
    raw = pd.read_csv("data/processed/sig_a_off/features_kalman.csv")
    kal = pd.read_csv("data/processed/sig_a/features_kalman.csv")
    stack = pd.read_csv("data/processed/sig_a/meta_features.csv")
    return {
        "y": raw["rop_mm_m"].to_numpy(),
        "raw": raw["rop_kf_tabpfn"].to_numpy(),
        "filtered": kal["rop_kf_tabpfn"].to_numpy(),
        "ckstack": stack["tabpfn"].to_numpy(),
        "rawstack": pd.read_csv(
            "data/processed/sig_a_off/meta_features.csv")["tabpfn"].to_numpy(),
    }


# Lithological units keyed by UCS. Every one of the eight distinct UCS values
# in the released file identifies exactly one unit of Table 1 of Gokceoglu
# (2021), which compiled them from the site investigation; the dataset carries
# the properties but not the unit names. The CAI printed there for slate (2.60)
# is a digit transposition of the 2.06 carried by both the data and Table 3 of
# Bardhan et al. (2024); UCS is unambiguous, so the units are keyed on it.
LITHOLOGY = {191.4: "Quartzite", 122.6: "Slate", 101.9: "Slate–metasandstone",
             101.0: "Quartzite–schist", 81.2: "Metasandstone",
             71.5: "Slate–metasandstone–schist", 46.0: "Metasandstone–schist",
             10.7: "Schist"}

# One hue per unit. Eight shades of a single hue proved impossible to tell
# apart, so the strip is a qualitative map and rock strength is left to the
# table. Hues are from Paul Tol's muted scheme, which is colourblind-safe, and
# are assigned so that no two units meeting along the tunnel share a hue
# family; the blue and orange of the prediction curves are deliberately absent.
LITHOLOGY_COLOUR = {191.4: "#882255", 122.6: "#117733", 101.9: "#DDCC77",
                    101.0: "#44AA99", 81.2: "#CC6677", 71.5: "#999933",
                    46.0: "#AA4499", 10.7: "#88CCEE"}


def fig_denoising(d, start=2750, width=200):
    """Three stacked panels over a single 300 m window.

    Superseded in the English manuscript by fig_windows, and kept because the
    frozen Turkish copy (paper/main-tr.tex) still includes its output.
    """
    sl = slice(start, start + width)
    x = np.arange(start, start + width) * SPACING_M
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.1), sharex=True, sharey=True)
    panels = [
        ("raw", ORANGE, "Unfiltered prediction series (level-1 signal)"),
        ("filtered", BLUE, "Kalman-filtered trend estimate (level-1 signal)"),
        ("ckstack", GREEN, "KDS output (level-2 prediction)"),
    ]
    for ax, (key, colour, label) in zip(axes, panels):
        ax.plot(x, d["y"][sl], color=GREY, lw=1.0, label="Measured ROP", zorder=1)
        ax.plot(x, d[key][sl], color=colour, lw=1.8, label=label, zorder=2)
        ax.set_ylabel("ROP (mm/min)")
        ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.2,
                  handlelength=1.6, borderaxespad=0.2, fontsize=7.5)
    top = max(d["y"][sl].max(), d["raw"][sl].max())
    axes[0].set_ylim(top=top * 1.30)
    axes[-1].set_xlabel("Distance along the TBM-driven section (m)")
    fig.subplots_adjust(hspace=0.12)
    fig.savefig(OUT / "fig_denoising.pdf")
    plt.close(fig)


def select_windows(y, ucs, width=200, parts=4):
    """Four 300 m windows, fixed by the geology rather than by model error.

    The drive is cut into four equal parts, and in each part the window is
    centred on the lithological boundary across which the mean measured ROP
    changes most. No prediction enters the choice, so the panels of
    fig_windows cannot have been picked for the size of the gain they happen
    to show.
    """
    n = len(y)
    bounds = np.flatnonzero(np.diff(ucs)) + 1
    starts = np.r_[0, bounds]
    block_mean = [y[lo:hi].mean() for lo, hi in zip(starts, np.r_[bounds, n])]

    def block_of(b):
        return int(np.searchsorted(starts, b, side="right") - 1)

    def jump(b):
        i = block_of(b)
        return abs(block_mean[i] - block_mean[i - 1])

    windows = []
    edges = np.linspace(0, n, parts + 1).astype(int)
    for p0, p1 in zip(edges[:-1], edges[1:]):
        inside = [b for b in bounds if p0 <= b < p1]
        if not inside:
            continue
        b = max(inside, key=jump)
        i = block_of(b)
        lo = min(max(0, b - width // 2), n - width)
        windows.append({"boundary": int(b), "lo": int(lo), "hi": int(lo + width),
                        "from": LITHOLOGY[ucs[starts[i - 1]]],
                        "to": LITHOLOGY[ucs[starts[i]]]})
    return windows


def window_gain(d, windows):
    """Mean absolute error of the two level-2 configurations in each window."""
    rows = []
    for w in windows:
        sl = slice(w["lo"], w["hi"])
        rows.append((mean_absolute_error(d["y"][sl], d["rawstack"][sl]),
                     mean_absolute_error(d["y"][sl], d["ckstack"][sl])))
    return rows


def report_window_gains(d, width=200):
    """The sentence in Section 4.2 that the Kalman gain is not local.

    Every complete non-overlapping window of the drive is scored under both
    level-2 configurations; the manuscript quotes the count, the mean and the
    range this prints.
    """
    n = len(d["y"])
    gains = []
    for lo in range(0, n - width + 1, width):
        sl = slice(lo, lo + width)
        gains.append(mean_absolute_error(d["y"][sl], d["rawstack"][sl])
                     - mean_absolute_error(d["y"][sl], d["ckstack"][sl]))
    gains = np.array(gains)
    print(f"window gains ({width} records = {width * SPACING_M:.0f} m): "
          f"{len(gains)} windows, {(gains > 0).sum()} favour filtering, "
          f"mean {gains.mean():+.3f}, range {gains.min():+.3f} to "
          f"{gains.max():+.3f} mm/min")
    return gains


def fig_windows(d, windows):
    """What the Kalman step buys, at four lithological boundaries.

    Both plotted predictions leave the same level-2 learner, TabPFN, given
    the same 53 engineered features and the same five prediction columns;
    they differ only in whether those columns were filtered first. The trend
    estimate itself is a level-1 input rather than a prediction of ROP, so it
    is not drawn here; the supplementary figure shows it over the whole
    section.
    """
    fig, axes = plt.subplots(len(windows), 1, figsize=(6.2, 7.2))
    gains = window_gain(d, windows)
    for k, (ax, w, (mae_u, mae_k)) in enumerate(zip(axes, windows, gains)):
        sl = slice(w["lo"], w["hi"])
        x = np.arange(w["lo"], w["hi"]) * SPACING_M
        ax.plot(x, d["y"][sl], color=GREY, lw=1.0, label="Measured ROP")
        ax.plot(x, d["rawstack"][sl], color=ORANGE, lw=1.3,
                label="Ensemble without filtering (level-2, TabPFN)")
        ax.plot(x, d["ckstack"][sl], color=GREEN, lw=1.6,
                label="KDS (level-2, TabPFN)")
        ax.axvline(w["boundary"] * SPACING_M, color="0.45", lw=0.8,
                   ls=(0, (4, 3)), zorder=0)
        ax.set_title(f"({'abcd'[k]}) {w['from']} to {w['to']} at "
                     f"{w['boundary'] * SPACING_M:,.0f} m"
                     f"      MAE {mae_u:.2f} → {mae_k:.2f} mm/min",
                     loc="left", fontsize=8)
        series = np.r_[d["y"][sl], d["rawstack"][sl], d["ckstack"][sl]]
        pad = 0.07 * (series.max() - series.min())
        ax.set_ylim(series.min() - pad, series.max() + pad)
        ax.set_ylabel("ROP (mm/min)")
    axes[-1].set_xlabel("Distance along the TBM-driven section (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.972),
               columnspacing=1.4, handlelength=1.6)
    fig.subplots_adjust(hspace=0.50, top=0.945)
    fig.savefig(OUT / "fig_windows.pdf")
    plt.close(fig)


def fig_longsection_full(d, windows):
    """The three series of the Kalman step over the whole TBM-driven section.

    fig_windows has to be read at 300 m to show individual records; the
    price is that a reader cannot tell whether the behaviour it shows is
    typical. This figure pays that back: the two candidate level-1 signals
    and the level-2 output at full resolution over all 5,334 records, with
    the lithological section above them and the four windows of fig_windows
    marked. It is a standalone A3 sheet, hence the wide canvas, and nothing
    here is smoothed.
    """
    n = len(d["y"])
    x = np.arange(n) * SPACING_M
    ucs = pd.read_csv("data/processed/sig_a/features_kalman.csv")[
        "ucs_mpa"].round(1).to_numpy()

    fig, axes = plt.subplots(
        4, 1, figsize=(14.6, 6.9), sharex=True,
        gridspec_kw={"height_ratios": [0.5, 3, 3, 3], "hspace": 0.11})
    axg, panels = axes[0], axes[1:]

    # --- lithological section -------------------------------------------------
    bounds = np.flatnonzero(np.diff(ucs)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, n]):
        axg.axvspan(x[lo], x[min(hi, n - 1)], color=LITHOLOGY_COLOUR[ucs[lo]],
                    lw=0)
    axg.set_ylim(0, 1)
    axg.set_yticks([])
    axg.set_ylabel("Lithology  ", fontsize=8, rotation=0, ha="right",
                   va="center")
    for sp in axg.spines.values():
        sp.set_visible(False)
    axg.tick_params(bottom=False)
    order = sorted(LITHOLOGY, reverse=True)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=LITHOLOGY_COLOUR[u], ec="0.4",
                             lw=0.4) for u in order]
    fig.legend(handles, [LITHOLOGY[u] for u in order], frameon=False,
               fontsize=8, ncol=8, loc="upper center",
               bbox_to_anchor=(0.5, 0.055), handlelength=1.2, handleheight=0.9,
               columnspacing=1.6, borderaxespad=0.0)

    # --- the three series -----------------------------------------------------
    # the labels name each series by its role in the ablation, because the two
    # level-1 panels are the two columns the arms of the experiment differ in
    spec = [("raw", ORANGE,
             "Unfiltered prediction series (level-1 column, ablation arm)"),
            ("filtered", BLUE,
             "Kalman-filtered trend estimate (level-1 column, KDS)"),
            ("ckstack", GREEN, "KDS output (level-2 prediction, TabPFN)")]
    for ax, (series, colour, label) in zip(panels, spec):
        ax.plot(x, d["y"], color=GREY, lw=0.35, label="Measured ROP", zorder=1)
        ax.plot(x, d[series], color=colour, lw=0.5, label=label, zorder=2)
        for w in windows:
            ax.axvspan(x[w["lo"]], x[w["hi"] - 1], color="0.15", alpha=0.10,
                       lw=0, zorder=0)
        ax.set_ylim(18, 104)
        ax.set_yticks([30, 50, 70, 90])
        ax.set_ylabel("ROP (mm/min)")
        leg = ax.legend(frameon=False, loc="upper left", ncol=2,
                        columnspacing=1.2, handlelength=1.6,
                        borderaxespad=0.2, fontsize=8)
        # the plotted lines are 0.5 pt wide so that 5,334 records stay legible;
        # the legend keys need to be thicker than that to show their colour
        for key in leg.get_lines():
            key.set_linewidth(1.8)
    for k, w in enumerate(windows):
        axg.annotate(f"({'abcd'[k]})",
                     xy=((x[w["lo"]] + x[w["hi"] - 1]) / 2, 1.35),
                     xycoords=("data", "axes fraction"), fontsize=8.5,
                     ha="center", va="bottom", color="0.25",
                     annotation_clip=False)
    panels[-1].set_xlabel("Distance along the TBM-driven section (m)")
    panels[-1].set_xlim(0, x[-1])
    fig.savefig(OUT / "fig_longsection_full.pdf")
    plt.close(fig)


def fig_accuracy(d):
    """Left: measured against predicted for the proposed method (the form
    conventional in this literature). Right: where the three configurations
    actually differ --- the distribution of absolute error."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))

    lim = (0, 125)
    p = d["ckstack"]
    ax = axes[0]
    ax.scatter(d["y"], p, s=2, alpha=0.18, color=BLUE, edgecolors="none", rasterized=True)
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Measured ROP (mm/min)")
    ax.set_ylabel("Predicted ROP (mm/min)")
    # The stored predictions are those of the reported configuration, the
    # pre-specified Q = 0.1, which is also the arm of every table.
    ax.set_title(f"KDS: $R^2$ = {r2_score(d['y'], p):.4f},\n"
                 f"MAE = {mean_absolute_error(d['y'], p):.2f} mm/min", fontsize=8.5)

    # The two non-proposed configurations nearly coincide, so the thick grey
    # curve is drawn first and the dashed orange one over it: the grey shows
    # through the gaps, which is itself the point.
    ax = axes[1]
    for key, colour, style, lw, name in [
        ("raw", GREY, "-", 2.4, "Best individual learner"),
        ("rawstack", ORANGE, (0, (4, 3)), 1.2, "Unfiltered ensemble"),
        ("ckstack", BLUE, "-", 1.8, "KDS"),
    ]:
        e = np.sort(np.abs(d["y"] - d[key]))
        ax.plot(e, np.arange(1, e.size + 1) / e.size * 100, color=colour,
                ls=style, lw=lw, label=name)
    med_c1 = np.median(np.abs(d["y"] - d["raw"]))
    med_c3 = np.median(np.abs(d["y"] - d["ckstack"]))
    ax.plot([med_c3, med_c1], [50, 50], color="black", lw=0.8, marker="|", ms=5)
    ax.annotate(f"median {med_c1:.2f} $\\to$ {med_c3:.2f}", (med_c1, 50),
                textcoords="offset points", xytext=(6, -3), fontsize=7.5, va="top")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Absolute error (mm/min)")
    ax.set_ylabel("Records within error (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax.set_title("Error distribution", fontsize=8.5)

    fig.subplots_adjust(wspace=0.35)
    fig.savefig(OUT / "fig_accuracy.pdf")
    plt.close(fig)


def fig_qsweep(d):
    """The filter's only free parameter against accuracy, with the two
    configurations it has to beat. Values are read from the stored sweep
    predictions rather than typed in, so the curve cannot drift from
    Table 8; the integrator is TabPFN at every Q, as in the reported
    configuration."""
    qs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    q = np.array(qs)
    r2 = np.array([
        r2_score(d["y"], pd.read_csv(
            f"data/processed/qs_{v}/meta_features.csv")["tabpfn"].to_numpy())
        for v in qs])
    unfiltered = r2_score(d["y"], d["rawstack"])
    single = r2_score(d["y"], d["raw"])

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.semilogx(q, r2, "o-", color=BLUE, lw=1.4, ms=5)
    ax.axhline(0.8637, color=GREEN, ls=":", lw=1.2)
    ax.axhline(single, color=ORANGE, ls="-.", lw=1.2)
    ax.axhline(unfiltered, color=GREY, ls="--", lw=1.2)
    ax.annotate(f"{r2[1]:.4f}", (q[1], r2[1]), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlim(0.007, 6.0)
    ax.set_ylim(0.8525, 0.8900)
    # Label the reference lines in place; a legend would sit on top of them.
    ax.text(5.6, 0.8643, "HENSM-RF (published)", ha="right", va="bottom",
            fontsize=8, color=GREEN)
    ax.text(0.0085, single - 0.0007, "Best individual learner", ha="left",
            va="top", fontsize=8, color=ORANGE)
    ax.text(5.6, unfiltered - 0.0007, "Unfiltered ensemble", ha="right",
            va="top", fontsize=8, color=GREY)
    ax.set_xlabel("Kalman process noise $Q$")
    ax.set_ylabel("$R^2$")
    fig.savefig(OUT / "fig_qsweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    ucs = pd.read_csv("data/processed/sig_a/features_kalman.csv")[
        "ucs_mpa"].round(1).to_numpy()
    windows = select_windows(data["y"], ucs)
    fig_windows(data, windows)
    report_window_gains(data)
    fig_denoising(data)
    fig_longsection_full(data, windows)
    fig_accuracy(data)
    fig_qsweep(data)
    print("wrote:", *(p.name for p in sorted(OUT.glob("fig_*.pdf"))))
