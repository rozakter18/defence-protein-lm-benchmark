"""Repeat the split-and-evaluate across seeds to get mean and spread.

Clustering is fixed; only the assignment of clusters to train/test changes.
This isolates split variance, which is large when clusters are few and uneven.
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


def random_split(df, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    nt, nv = int(len(df) * TEST_FRACTION), int(len(df) * VAL_FRACTION)
    out = np.array(["train"] * len(df), dtype=object)
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
    members = read_clusters(INTERIM / f"mmseqs_{CLUSTER_IDENTITY}" / "clu_cluster.tsv")

    seqs = df.sequence.tolist()
    feats = {
        "esm2": np.vstack([cache[p] for p in df.protein_id]),
        "aac": aac_features(seqs),
        "kmer": kmer_features(seqs),
    }
    y = df.system_type.values

    rows = []
    for seed in range(N_SEEDS):
        splits = {"clustered": clustered_split(df, members, seed),
                  "random": random_split(df, seed)}
        for sname, sp in splits.items():
            for fname, X in feats.items():
                rows.append({"seed": seed, "split": sname, "features": fname,
                             "macro_f1": score(X, y, sp)})
        print(f"seed {seed} done")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"repeats_id{CLUSTER_IDENTITY}.csv", index=False)

    summary = (out.groupby(["features", "split"])["macro_f1"]
                  .agg(["mean", "std", "min", "max"]).round(3))
    print(f"\n{summary.to_string()}")

    gap = out.pivot_table(index=["features", "seed"], columns="split",
                          values="macro_f1")
    gap["leak"] = gap["random"] - gap["clustered"]
    print("\nleakage gap per feature set:")
    print(gap.groupby("features")["leak"].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
