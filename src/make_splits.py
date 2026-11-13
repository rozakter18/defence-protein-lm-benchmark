"""Build both split types.

random     — shuffle sequences, split. Homologs land on both sides.
clustered  — MMseqs2 cluster at 30% identity, assign whole clusters to one side.

Clusters are assigned to test greedily from largest to smallest until the test
fraction is reached, which keeps class balance closer than random cluster
assignment when cluster sizes are very uneven.
"""

import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from config import (
    CLUSTER_COVERAGE,
    CLUSTER_IDENTITY,
    INTERIM,
    SEED,
    TEST_FRACTION,
    VAL_FRACTION,
)


def write_fasta(df, path):
    with open(path, "w") as fh:
        for row in df.itertuples():
            fh.write(f">{row.protein_id}\n{row.sequence}\n")


def run_mmseqs(fasta, outdir):
    if not shutil_which("mmseqs"):
        sys.exit(
            "mmseqs not found on PATH.\n"
            "  conda install -c bioconda mmseqs2 -y\n"
            "or grab the static binary from github.com/soedinglab/MMseqs2/releases"
        )
    outdir.mkdir(exist_ok=True)
    prefix = outdir / "clu"
    cmd = [
        "mmseqs", "easy-cluster", str(fasta), str(prefix), str(outdir / "tmp"),
        "--min-seq-id", str(CLUSTER_IDENTITY),
        "-c", str(CLUSTER_COVERAGE),
        "--cov-mode", "0",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return prefix.with_name("clu_cluster.tsv")


def shutil_which(name):
    from shutil import which
    return which(name)


def read_clusters(tsv):
    members = defaultdict(list)
    with open(tsv) as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            members[rep].append(member)
    return members


def clustered_split(df, members):
    n_test = int(len(df) * TEST_FRACTION)
    n_val = int(len(df) * VAL_FRACTION)

    order = sorted(members.items(), key=lambda kv: -len(kv[1]))
    assignment, filled_test, filled_val = {}, 0, 0
    for _, mem in order:
        if filled_test < n_test:
            split, filled_test = "test", filled_test + len(mem)
        elif filled_val < n_val:
            split, filled_val = "val", filled_val + len(mem)
        else:
            split = "train"
        for m in mem:
            assignment[m] = split
    return df["protein_id"].map(assignment).fillna("train")


def random_split(df, rng):
    idx = rng.permutation(len(df))
    n_test = int(len(df) * TEST_FRACTION)
    n_val = int(len(df) * VAL_FRACTION)
    out = np.array(["train"] * len(df), dtype=object)
    out[idx[:n_test]] = "test"
    out[idx[n_test : n_test + n_val]] = "val"
    return pd.Series(out, index=df.index)


def main():
    path = INTERIM / "dataset.parquet"
    if not path.exists():
        path = INTERIM / "defence_proteins.parquet"
        print(f"dataset.parquet not found, using {path.name} (Task A only)")
    df = pd.read_parquet(path)

    fasta = INTERIM / "all.fasta"
    write_fasta(df, fasta)

    tsv = run_mmseqs(fasta, INTERIM / "mmseqs")
    members = read_clusters(tsv)
    print(f"{len(members):,} clusters from {len(df):,} sequences")

    df["split_clustered"] = clustered_split(df, members)
    df["split_random"] = random_split(df, np.random.default_rng(SEED))

    for col in ("split_clustered", "split_random"):
        print(f"\n{col}:\n{df[col].value_counts().to_string()}")

    df.to_parquet(INTERIM / "dataset_split.parquet", index=False)
    print(f"\nwrote {INTERIM / 'dataset_split.parquet'}")


if __name__ == "__main__":
    main()
