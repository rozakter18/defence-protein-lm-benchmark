"""Download DefenseFinder's precomputed RefSeq annotations.

The exact download URL and column names on the DefenseFinder site change
between releases. Confirm both against https://defensefinder.mdmlab.fr/wiki/refseq/
before the first run and update DEFENSEFINDER_URL / COLUMN_MAP below if needed.

If the automatic download fails, download the table manually into
data/raw/ and re-run — this script will use an existing file.
"""

import sys

import pandas as pd
import requests
from tqdm import tqdm

from config import RAW, INTERIM, N_SYSTEM_TYPES, MIN_SEQS_PER_TYPE

DEFENSEFINDER_URL = (
    "https://defensefinder.mdmlab.fr/static/refseq/defense_finder_genes.tsv"
)
LOCAL = RAW / "defense_finder_genes.tsv"

# Maps whatever the release calls these onto the names used downstream.
COLUMN_MAP = {
    "hit_id": "protein_id",
    "replicon": "genome_id",
    "type": "system_type",
    "subtype": "system_subtype",
}


def download(url, dest):
    if dest.exists():
        print(f"using existing {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, disable=total == 0
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))


def main():
    try:
        download(DEFENSEFINDER_URL, LOCAL)
    except Exception as exc:
        sys.exit(
            f"download failed: {exc}\n"
            f"Fetch the gene table manually from "
            f"https://defensefinder.mdmlab.fr/wiki/refseq/ and save it to {LOCAL}"
        )

    df = pd.read_csv(LOCAL, sep="\t", low_memory=False)
    print(f"loaded {len(df):,} rows, columns: {list(df.columns)}")

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        sys.exit(
            f"expected columns not found: {missing}\n"
            f"Actual columns: {list(df.columns)}\n"
            f"Update COLUMN_MAP in this file to match."
        )

    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].dropna()
    df = df.drop_duplicates(subset="protein_id")

    counts = df["system_type"].value_counts()
    keep = counts[counts >= MIN_SEQS_PER_TYPE].head(N_SYSTEM_TYPES).index
    df = df[df["system_type"].isin(keep)].reset_index(drop=True)

    print(f"\nkeeping {len(keep)} system types, {len(df):,} proteins")
    print(df["system_type"].value_counts().to_string())

    out = INTERIM / "annotations.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
