"""Per-class reports and confusion matrices at every clustering threshold.

Reads each clustering directory found under data/interim/, rebuilds the
stratified clustered split with a fixed seed, trains logistic regression on the
cached ESM embeddings, and writes a report plus a confusion-matrix heatmap for
each threshold so the two can be compared side by side.
"""

import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from config import INTERIM, RESULTS, SEED, TEST_FRACTION, VAL_FRACTION

INK = "#3F345B"
MUTED = "#6A5A8F"
RAMP = LinearSegmentedColormap.from_list("nature", ["#F6F9FC", "#DDE6F3",
                                                    "#B59BC8", "#6A5A8F",
                                                    "#3F345B"])


def find_clusterings():
    """Return {label: path to clu_cluster.tsv}, one per threshold on disk."""
    found = {}
    for d in sorted(INTERIM.glob("mmseqs*")):
        tsv = d / "clu_cluster.tsv"
        if not tsv.exists():
            continue
        m = re.search(r"mmseqs_?([\d.]+)?$", d.name)
        ident = m.group(1) if m and m.group(1) else "0.3"
        found[f"{int(float(ident) * 100)}%"] = tsv
    return found


def read_clusters(tsv):
    m = defaultdict(list)
    for line in open(tsv):
        rep, member = line.rstrip("\n").split("\t")
        m[rep].append(member)
    return m


def clustered_split(df, members, seed):
    rng = np.random.default_rng(seed)
    pid_to_type = dict(zip(df.protein_id, df.system_type))
    known = set(pid_to_type)
    by_type = defaultdict(list)
    for mem in members.values():
        mem = [m for m in mem if m in known]
        if mem:
            maj = Counter(pid_to_type[m] for m in mem).most_common(1)[0][0]
            by_type[maj].append(mem)
    out = {}
    for groups in by_type.values():
        total = sum(len(m) for m in groups)
        wt, wv = total * TEST_FRACTION, total * VAL_FRACTION
        nt = nv = 0
        for i in rng.permutation(len(groups)):
            mem = groups[i]
            if nt < wt:
                sp, nt = "test", nt + len(mem)
            elif nv < wv:
                sp, nv = "val", nv + len(mem)
            else:
                sp = "train"
            for m in mem:
                out[m] = sp
    return df.protein_id.map(out).fillna("train").values


def plot_confusion(cm, labels, title, path):
    norm = cm / cm.sum(axis=1, keepdims=True)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3.2, 0.62 * n + 2.4), dpi=200)
    ax.imshow(norm, cmap=RAMP, vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)

    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=9,
                  color=INK)
    ax.set_yticks(range(n), labels, fontsize=9, color=INK)
    ax.set_xlabel("Predicted", fontsize=10, color=INK)
    ax.set_ylabel("True", fontsize=10, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    df = pd.read_parquet(INTERIM / "dataset_split.parquet")
    cache = dict(np.load(INTERIM / "embeddings.npz"))
    df = df[df.protein_id.isin(cache)].reset_index(drop=True)
    X = np.vstack([cache[p] for p in df.protein_id])
    y = df.system_type.values
    labels = sorted(set(y))

    clusterings = find_clusterings()
    if not clusterings:
        raise SystemExit(f"no clustering directories found under {INTERIM}")
    print(f"found clusterings: {', '.join(clusterings)}\n")

    for label, tsv in clusterings.items():
        members = read_clusters(tsv)
        split = clustered_split(df, members, SEED)
        tr, te = split == "train", split == "test"

        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))

        macro = f1_score(y[te], pred, average="macro")
        print(f"=== {label} identity — {len(members):,} clusters, "
              f"{te.sum():,} test, macro-F1 {macro:.3f}")
        print(classification_report(y[te], pred, digits=3, zero_division=0))

        rep = pd.DataFrame(classification_report(
            y[te], pred, output_dict=True, zero_division=0)).T
        rep.to_csv(RESULTS / f"report_id{label.rstrip('%')}.csv")

        cm = confusion_matrix(y[te], pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            RESULTS / f"confusion_id{label.rstrip('%')}.csv")
        plot_confusion(
            cm, labels,
            f"Confusion at {label} identity clustering  ·  "
            f"macro-F1 {macro:.3f}\nESM-2 embeddings, logistic regression, "
            f"counts shown, shaded by row fraction",
            RESULTS / f"confusion_id{label.rstrip('%')}.png")
        print(f"wrote report_id{label.rstrip('%')}.csv, "
              f"confusion_id{label.rstrip('%')}.csv/.png\n")


if __name__ == "__main__":
    main()
