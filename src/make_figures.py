"""Figures for the README.

Reads results/repeats_id*.csv (produced by repeat_eval.py) and draws the
leakage comparison: macro-F1 under random vs clustered splits for each feature
set, with standard deviation across cluster-assignment seeds.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLUSTER_IDENTITY, RESULTS

# Palette adapted from a Nature cover scheme. The violet/green pair is chosen
# over the scheme's blue/pink, which is indistinguishable under red-green
# colour blindness (CVD delta-E 3.9, well below the usable floor of 8).
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


def main():
    path = RESULTS / f"repeats_id{CLUSTER_IDENTITY}.csv"
    if not path.exists():
        found = sorted(RESULTS.glob("repeats_id*.csv"))
        if not found:
            sys.exit(f"No repeats_id*.csv in {RESULTS}. Run repeat_eval.py first.")
        path = found[-1]
    df = pd.read_csv(path)
    print(f"reading {path.name}")

    stats = (df.groupby(["features", "split"])["macro_f1"]
               .agg(["mean", "std"]).reset_index())
    order = ["esm2", "kmer", "aac"]
    stats["features"] = pd.Categorical(stats.features, order, ordered=True)
    stats = stats.sort_values("features")

    x = np.arange(len(order))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)

    for i, (split, colour) in enumerate([("random", RANDOM_C),
                                         ("clustered", CLUSTER_C)]):
        sub = stats[stats.split == split].set_index("features").loc[order]
        off = (i - 0.5) * (w + 0.015)
        bars = ax.bar(x + off, sub["mean"], w, yerr=sub["std"], capsize=3,
                      color=colour, edgecolor="none",
                      error_kw=dict(ecolor=MUTED, lw=1.1),
                      label=f"{split} split")
        for b, m, s in zip(bars, sub["mean"], sub["std"]):
            ax.text(b.get_x() + b.get_width() / 2, m + s + 0.022,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=9, color=INK)

    # Gap annotation above each pair, on a common baseline.
    for xi, feat in zip(x, order):
        sub = stats[stats.features == feat].set_index("split")
        gap = sub.loc["random", "mean"] - sub.loc["clustered", "mean"]
        big = gap > 0.1
        ax.text(xi, 1.115, f"gap {gap:.3f}", ha="center", va="center",
                fontsize=10, color=CLUSTER_C if big else MUTED,
                fontweight="bold" if big else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in order], fontsize=10, color=INK)
    ax.set_ylabel("Macro-F1 on held-out test set", fontsize=10, color=INK)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.legend(frameon=False, fontsize=10, loc="upper right",
              bbox_to_anchor=(1.0, 0.87), labelcolor=INK, handlelength=1.2)
    ax.set_title(
        f"Removing homologs costs the cheap baselines far more than the "
        f"embeddings\nDefence system family classification, "
        f"{int(CLUSTER_IDENTITY * 100)}% identity clustering, "
        f"mean ± s.d. over 10 seeds",
        fontsize=10.5, color=INK, loc="left", pad=14)

    fig.tight_layout()
    out = RESULTS / "leakage_gap.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
