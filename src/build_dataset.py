"""Fetch protein sequences for the annotated defence proteins, plus a matched
set of background proteins for Task B.

Sequences come from NCBI Entrez in batches. With an API key this runs at about
10 requests/second; 20k proteins in batches of 200 is ~100 requests, so a few
minutes. Without a key it is roughly three times slower.

Background proteins: for a sample of the same genomes, every protein that
DefenseFinder did not flag is a candidate, and we sample from those to match
the defence set size. They are assumed non-defence, which is an assumption and
not a fact — see the README limitations.
"""

import time

import pandas as pd
from Bio import Entrez, SeqIO
from tqdm import tqdm

from config import (
    ENTREZ_API_KEY,
    ENTREZ_EMAIL,
    INTERIM,
    MAX_SEQ_LEN,
    SEED,
)

Entrez.email = ENTREZ_EMAIL
if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY

BATCH = 200


def fetch_sequences(accessions):
    """Return {accession: sequence} for a list of protein accessions."""
    seqs = {}
    accessions = list(dict.fromkeys(accessions))
    for i in tqdm(range(0, len(accessions), BATCH), desc="entrez"):
        chunk = accessions[i : i + BATCH]
        for attempt in range(4):
            try:
                handle = Entrez.efetch(
                    db="protein", id=",".join(chunk), rettype="fasta", retmode="text"
                )
                for rec in SeqIO.parse(handle, "fasta"):
                    acc = rec.id.split("|")[-1] if "|" in rec.id else rec.id
                    seqs[acc] = str(rec.seq)
                handle.close()
                break
            except Exception as exc:
                wait = 2 ** attempt
                print(f"  batch {i} failed ({exc}), retrying in {wait}s")
                time.sleep(wait)
        else:
            print(f"  giving up on batch {i}")
        time.sleep(0.1 if ENTREZ_API_KEY else 0.35)
    return seqs


def clean(df):
    df = df[df["sequence"].str.len().between(50, 10000)]
    df = df[~df["sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)]
    df["sequence"] = df["sequence"].str.slice(0, MAX_SEQ_LEN)
    return df.reset_index(drop=True)


def main():
    ann = pd.read_parquet(INTERIM / "annotations.parquet")
    print(f"{len(ann):,} annotated defence proteins")

    seqs = fetch_sequences(ann["protein_id"].tolist())
    print(f"retrieved {len(seqs):,} sequences")

    ann["sequence"] = ann["protein_id"].map(seqs)
    ann = ann.dropna(subset=["sequence"])
    ann = clean(ann)
    ann["label_binary"] = "defence"

    print(f"{len(ann):,} defence proteins after cleaning")
    ann.to_parquet(INTERIM / "defence_proteins.parquet", index=False)

    print(
        "\nBackground set for Task B is not fetched automatically.\n"
        "Download protein FASTA for a sample of the genomes in\n"
        "  data/interim/defence_proteins.parquet (column genome_id)\n"
        "using the NCBI datasets CLI:\n"
        "  datasets download genome accession <ACC> --include protein\n"
        "then run build_background.py. Sampling whole genomes keeps the\n"
        "background compositionally matched to the defence set, which matters:\n"
        "if the background came from unrelated taxa the classifier could\n"
        "separate the classes on GC-driven amino-acid bias alone."
    )


if __name__ == "__main__":
    main()
