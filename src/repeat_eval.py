"""Repeat split-and-evaluate across seeds, for every task, to get mean and spread.

Clustering is fixed; only the assignment of clusters to train/test changes.
This isolates split variance, which is large when clusters are few and uneven.

Task A (family) uses only rows with a system_type. Task B (detection) uses
everything, including the background proteins.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from config import CLUSTER_IDENTITY, INTERIM, RESULTS, TEST_FRACTION, VAL_FRACTION
from train_eval import aac_features, kmer_features

N_SEEDS = 10


def read_clusters(tsv):
    m = defaultdict(list)
    for line in open(tsv):
        rep, member = line.rstrip("\n").split("\t")
        m[rep].append(member)
    return m


def clustered_split(ids, strat, members, seed):
    """ids/strat are aligned arrays for the rows being split."""
    rng = np.random.default_rng(seed)
    pid_to_type = dict(zip(ids, strat))
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
    return np.array([out.get(i, "train") for i in ids], dtype=object)


def random_split(n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nt, nv = int(n * TEST_FRACTION), int(n * VAL_FRACTION)
    out = np.array(["train"] * n, dtype=object)
    out[idx[:nt]] = "test"
    out[idx[nt:nt + nv]] = "val"
    return out


def score(X, y, split):
    tr, te = split == "train", split == "test"
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
    return f1_score(y[te], clf.predict(sc.transform(X[te])), average="macro")


def main():
    df = pd.read_parquet(INTERIM / "dataset_split.parquet")
    cache = dict(np.load(INTERIM / "embeddings.npz"))
    df = df[df.protein_id.isin(cache)].reset_index(drop=True)
    members = read_clusters(
        INTERIM / f"mmseqs_{CLUSTER_IDENTITY}" / "clu_cluster.tsv")

    tasks = {}
    if df.system_type.notna().any():
        tasks["A_family"] = df.system_type.notna().values
    if "label_binary" in df.columns and df.label_binary.nunique() > 1:
        tasks["B_detection"] = np.ones(len(df), bool)

    rows = []
    for task, mask in tasks.items():
        sub = df[mask].reset_index(drop=True)
        seqs = sub.sequence.tolist()
        feats = {
            "esm2": np.vstack([cache[p] for p in sub.protein_id]),
            "aac": aac_features(seqs),
            "kmer": kmer_features(seqs),
        }
        if task == "A_family":
            y, strat = sub.system_type.values, sub.system_type.values
        else:
            y = sub.label_binary.values
            strat = sub.system_type.fillna(sub.label_binary).values
        ids = sub.protein_id.values
        print(f"\n{task}: {len(sub):,} rows, {len(set(y))} classes")

        for seed in range(N_SEEDS):
            splits = {
                "clustered": clustered_split(ids, strat, members, seed),
                "random": random_split(len(sub), seed),
            }
            for sname, sp in splits.items():
                for fname, X in feats.items():
                    rows.append({"task": task, "seed": seed, "split": sname,
                                 "features": fname,
                                 "macro_f1": score(X, y, sp)})
            print(f"  seed {seed} done")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"repeats_id{CLUSTER_IDENTITY}.csv", index=False)

    summary = (out.groupby(["task", "features", "split"])["macro_f1"]
                  .agg(["mean", "std"]).round(3))
    print(f"\n{summary.to_string()}")

    gap = out.pivot_table(index=["task", "features", "seed"], columns="split",
                          values="macro_f1")
    gap["leak"] = gap["random"] - gap["clustered"]
    print("\nleakage gap:")
    print(gap.groupby(["task", "features"])["leak"]
             .agg(["mean", "std"]).round(3).to_string())
    print(f"\nwrote {RESULTS / f'repeats_id{CLUSTER_IDENTITY}.csv'}")


if __name__ == "__main__":
    main()
