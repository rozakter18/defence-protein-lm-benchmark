"""Figures for the README.

Reads results/repeats_id*.csv (from repeat_eval.py) and draws one panel per
task: macro-F1 under random vs clustered splits for each feature set, with
standard deviation across cluster-assignment seeds.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLUSTER_IDENTITY, RESULTS

# Palette adapted from a Nature cover. The scheme's blue and pink are not used
# together: their separation under red-green colour blindness is delta-E 3.9,
# below the usable floor of 8.
RANDOM_C = "#5E4B87"
CLUSTER_C = "#5E9173"
INK = "#3F345B"
MUTED = "#6A5A8F"
GRID = "#DDE6F3"

LABELS = {
    "esm2": "ESM-2 150M\nembeddings",
    "kmer": "3-mer\ncomposition",
    "aac": "Amino-acid\ncomposition",
}
TITLES = {
    "A_family": "Task A — defence system family\n(12 classes, homologous)",
    "B_detection": "Task B — defence vs background\n(2 classes, non-homologous)",
}
ORDER = ["esm2", "kmer", "aac"]


def panel(ax, stats, title, show_legend):
    x = np.arange(len(ORDER))
    w = 0.34
    for i, (split, colour) in enumerate([("random", RANDOM_C),
                                         ("clustered", CLUSTER_C)]):
        sub = stats[stats.split == split].set_index("features").loc[ORDER]
        off = (i - 0.5) * (w + 0.015)
        bars = ax.bar(x + off, sub["mean"], w, yerr=sub["std"], capsize=3,
                      color=colour, edgecolor="none",
                      error_kw=dict(ecolor=MUTED, lw=1.1),
                      label=f"{split} split")
        for b, m, s in zip(bars, sub["mean"], sub["std"]):
            ax.text(b.get_x() + b.get_width() / 2, m + s + 0.022, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)

    for xi, feat in zip(x, ORDER):
        sub = stats[stats.features == feat].set_index("split")
        gap = sub.loc["random", "mean"] - sub.loc["clustered", "mean"]
        sd = np.hypot(sub.loc["random", "std"], sub.loc["clustered", "std"])
        # Emphasise by magnitude, not significance: a gap can be reliably
        # non-zero and still be too small to matter.
        big = gap > 0.1
        ax.text(xi, 1.135, f"{gap:.3f}", ha="center", va="center", fontsize=9.5,
                color=CLUSTER_C if big else MUTED,
                fontweight="bold" if big else "normal")
        ax.text(xi, 1.065, "± " + f"{sd:.3f}", ha="center", va="center",
                fontsize=8, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in ORDER], fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
    if show_legend:
        ax.legend(frameon=False, fontsize=9.5, loc="upper right",
                  bbox_to_anchor=(1.0, 0.80), labelcolor=INK, handlelength=1.2)


def main():
    path = RESULTS / f"repeats_id{CLUSTER_IDENTITY}.csv"
    if not path.exists():
        found = sorted(RESULTS.glob("repeats_id*.csv"))
        if not found:
            sys.exit(f"No repeats_id*.csv in {RESULTS}. Run repeat_eval.py first.")
        path = found[-1]
    df = pd.read_csv(path)
    print(f"reading {path.name}")

    if "task" not in df.columns:
        df["task"] = "A_family"

    tasks = [t for t in ["A_family", "B_detection"] if t in set(df.task)]
    stats = (df.groupby(["task", "features", "split"])["macro_f1"]
               .agg(["mean", "std"]).reset_index())

    fig, axes = plt.subplots(1, len(tasks), figsize=(6.6 * len(tasks), 4.8),
                             dpi=200, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, task in zip(axes, tasks):
        panel(ax, stats[stats.task == task], TITLES.get(task, task),
              show_legend=(ax is axes[0]))
    axes[0].set_ylabel("Macro-F1 on held-out test set", fontsize=10, color=INK)

    fig.suptitle(
        "Homology leakage appears where homology does\n"
        f"Numbers above each pair are the random-minus-clustered gap, "
        f"± combined s.d. over 10 seeds; {int(CLUSTER_IDENTITY * 100)}% "
        f"identity clustering",
        fontsize=11, color=INK, x=0.012, ha="left", y=1.06)

    fig.tight_layout()
    out = RESULTS / "leakage_gap.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
