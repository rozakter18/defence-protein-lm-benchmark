"""Build both split types.

random     - shuffle sequences, split. Homologs land on both sides.
clustered  - MMseqs2 cluster at 30% identity, whole clusters to one side,
             assigned stratified by system type so every class is represented
             in train, val and test.

A cluster spanning several system types is balanced under its majority type,
so it is counted once rather than for each.
"""

import subprocess
import sys
from collections import Counter, defaultdict
from shutil import which

import numpy as np
import pandas as pd

from config import (CLUSTER_COVERAGE, CLUSTER_IDENTITY, INTERIM, SEED,
                    TEST_FRACTION, VAL_FRACTION)


def write_fasta(df, path):
    with open(path, "w") as fh:
        for row in df.itertuples():
            fh.write(f">{row.protein_id}\n{row.sequence}\n")


def run_mmseqs(fasta, outdir):
    tsv = outdir / "clu_cluster.tsv"
    if tsv.exists():
        print(f"reusing existing clustering: {tsv.name}")
        return tsv
    if not which("mmseqs"):
        sys.exit("mmseqs not found on PATH.")
    outdir.mkdir(exist_ok=True)
    cmd = ["mmseqs", "easy-cluster", str(fasta), str(outdir / "clu"),
           str(outdir / "tmp"), "--min-seq-id", str(CLUSTER_IDENTITY),
           "-c", str(CLUSTER_COVERAGE), "--cov-mode", "0",
           "-s", "7.5"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return tsv


def read_clusters(tsv):
    members = defaultdict(list)
    with open(tsv) as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            members[rep].append(member)
    return members


def clustered_split(df, members):
    rng = np.random.default_rng(SEED)
    pid_to_type = dict(zip(df["protein_id"], df["system_type"]))
    known = set(pid_to_type)

    by_type = defaultdict(list)
    for rep, mem in members.items():
        mem = [m for m in mem if m in known]
        if not mem:
            continue
        majority = Counter(pid_to_type[m] for m in mem).most_common(1)[0][0]
        by_type[majority].append(mem)

    assignment = {}
    for system_type, groups in by_type.items():
        total = sum(len(m) for m in groups)
        want_test, want_val = total * TEST_FRACTION, total * VAL_FRACTION
        n_test = n_val = 0
        for i in rng.permutation(len(groups)):
            mem = groups[i]
            if n_test < want_test:
                split, n_test = "test", n_test + len(mem)
            elif n_val < want_val:
                split, n_val = "val", n_val + len(mem)
            else:
                split = "train"
            for m in mem:
                assignment[m] = split

    return df["protein_id"].map(assignment).fillna("train")


def random_split(df):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * TEST_FRACTION)
    n_val = int(len(df) * VAL_FRACTION)
    out = np.array(["train"] * len(df), dtype=object)
    out[idx[:n_test]] = "test"
    out[idx[n_test:n_test + n_val]] = "val"
    return pd.Series(out, index=df.index)


def main():
    path = INTERIM / "dataset.parquet"
    if not path.exists():
        path = INTERIM / "defence_proteins.parquet"
        print(f"using {path.name} (Task A only)")
    df = pd.read_parquet(path)

    fasta = INTERIM / "all.fasta"
    write_fasta(df, fasta)
    members = read_clusters(run_mmseqs(fasta, INTERIM / f"mmseqs_{CLUSTER_IDENTITY}"))
    print(f"{len(members):,} clusters from {len(df):,} sequences")

    df["split_clustered"] = clustered_split(df, members)
    df["split_random"] = random_split(df)

    print("\nclustered split by class:")
    print(pd.crosstab(df["system_type"], df["split_clustered"]).to_string())

    leak = sum(1 for mem in members.values()
               if len(set(df.set_index("protein_id")["split_clustered"]
                          .reindex(mem).dropna())) > 1)
    print(f"\nclusters spanning >1 split (should be 0): {leak}")

    df.to_parquet(INTERIM / "dataset_split.parquet", index=False)
    print(f"wrote {INTERIM / 'dataset_split.parquet'}")


if __name__ == "__main__":
    main()
