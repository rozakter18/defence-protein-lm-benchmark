"""Parse the DefenseFinder RefSeq export into one row per protein.

Download the table from https://defensefinder.mdmlab.fr/wiki/refseq/ (the red
"RefSeq" download link) and drop it in data/raw/. This script finds it, works
out the delimiter, maps the columns, and caps each system type so classes stay
balanced and the embedding pass stays tractable.
"""

import sys

import pandas as pd

from config import (INTERIM, MAX_SEQS_PER_TYPE, MIN_SEQS_PER_TYPE,
                    N_SYSTEM_TYPES, RAW)

COLUMN_MAP = {
    "accession_in_sys": "protein_id",
    "accessions": "protein_id",
    "system": "system_type",
    "type": "system_type",
    "subsystem": "system_subtype",
    "subtype": "system_subtype",
    "replicon": "genome_id",
    "assembly": "assembly",
    "species": "species",
    "genus": "genus",
    "phylum": "phylum",
}

ACCESSION_SPLIT = r"[,;|\s]+"


def find_input():
    candidates = sorted(
        p for p in RAW.iterdir()
        if p.suffix.lower() in {".tsv", ".csv", ".txt"} and p.stat().st_size > 1000
    )
    if not candidates:
        sys.exit(f"No table found in {RAW}. Download the RefSeq export first.")
    if len(candidates) > 1:
        print(f"found {len(candidates)} files, using {candidates[0].name}")
    return candidates[0]


def load(path):
    head = path.open(encoding="utf-8", errors="replace").readline()
    sep = "\t" if head.count("\t") > head.count(",") else ","
    print(f"reading {path.name} with sep={sep!r}")
    return pd.read_csv(path, sep=sep, low_memory=False)


def main():
    df = load(find_input())
    print(f"{len(df):,} rows\n")

    rename = {c: COLUMN_MAP[c.strip().lower()] for c in df.columns
              if c.strip().lower() in COLUMN_MAP}
    df = df.rename(columns=rename)

    missing = {"protein_id", "system_type"} - set(df.columns)
    if missing:
        sys.exit(f"could not map: {missing}\nColumns were: {list(df.columns)}")

    keep = [c for c in ["protein_id", "system_type", "system_subtype",
                        "genome_id", "assembly", "species", "genus", "phylum"]
            if c in df.columns]
    df = df[keep].dropna(subset=["protein_id", "system_type"])

    before = len(df)
    df["protein_id"] = df["protein_id"].astype(str).str.strip()
    df = df.assign(protein_id=df["protein_id"].str.split(ACCESSION_SPLIT))
    df = df.explode("protein_id")
    df["protein_id"] = df["protein_id"].str.strip()
    df = df[df["protein_id"].str.len() > 3]
    df = df.drop_duplicates(subset="protein_id").reset_index(drop=True)
    print(f"{before:,} rows -> {len(df):,} unique proteins")

    counts = df["system_type"].value_counts()
    eligible = counts[counts >= MIN_SEQS_PER_TYPE]
    if eligible.empty:
        sys.exit(f"No type has {MIN_SEQS_PER_TYPE}+ proteins. "
                 f"Largest: {counts.index[0]} ({counts.iloc[0]}).")

    keep_types = eligible.head(N_SYSTEM_TYPES).index
    df = df[df["system_type"].isin(keep_types)].reset_index(drop=True)

    if MAX_SEQS_PER_TYPE:
        parts = [g.sample(min(len(g), MAX_SEQS_PER_TYPE), random_state=0)
                 for _, g in df.groupby("system_type")]
        df = pd.concat(parts).reset_index(drop=True)
        print(f"capped at {MAX_SEQS_PER_TYPE} per type")

    print(f"\n{len(keep_types)} system types, {len(df):,} proteins\n")
    print(df["system_type"].value_counts().to_string())

    out = INTERIM / "annotations.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
