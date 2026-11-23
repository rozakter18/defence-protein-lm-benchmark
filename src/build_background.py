"""Build the background (non-defence) protein set for Task B.

Background proteins are sampled from the same genomes the defence proteins came
from. That matters: drawn from unrelated taxa, the two classes could be
separated on GC-driven amino-acid bias alone, and the task would measure
nothing.

Needs the NCBI datasets CLI:
    conda install -c conda-forge ncbi-datasets-cli -y

Writes data/interim/dataset.parquet, which make_splits.py picks up in
preference to defence_proteins.parquet.
"""

import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import INTERIM, MAX_SEQ_LEN, RAW, SEED

# How many background proteins to collect. Match the defence count for a
# balanced task; lower it if the embedding pass has to run on CPU.
N_BACKGROUND = 8000

# Genomes to download. Each yields a few thousand proteins, so this is
# comfortably more than needed; extras are simply not sampled.
N_GENOMES = 60

DOWNLOAD_DIR = RAW / "genomes"


def check_cli():
    if not shutil.which("datasets"):
        sys.exit(
            "NCBI datasets CLI not found.\n"
            "  conda install -c conda-forge ncbi-datasets-cli -y"
        )


def download_genomes(accessions):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    got = []
    for acc in tqdm(accessions, desc="genomes"):
        out = DOWNLOAD_DIR / f"{acc}.zip"
        if out.exists() and out.stat().st_size > 1000:
            got.append(out)
            continue
        r = subprocess.run(
            ["datasets", "download", "genome", "accession", acc,
             "--include", "protein", "--filename", str(out)],
            capture_output=True, text=True)
        if r.returncode == 0 and out.exists():
            got.append(out)
        else:
            tqdm.write(f"  skipped {acc}: {r.stderr.strip()[:90]}")
    return got


def read_proteins(zip_path):
    """Yield (accession, sequence) from the protein.faa inside a datasets zip."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if n.endswith("protein.faa")]
            if not names:
                return
            with z.open(names[0]) as fh:
                acc, chunks = None, []
                for raw in fh:
                    line = raw.decode("utf-8", "replace").rstrip()
                    if line.startswith(">"):
                        if acc:
                            yield acc, "".join(chunks)
                        acc = line[1:].split()[0]
                        chunks = []
                    else:
                        chunks.append(line)
                if acc:
                    yield acc, "".join(chunks)
    except zipfile.BadZipFile:
        tqdm.write(f"  corrupt zip: {zip_path.name}")


def clean(df):
    df = df[df.sequence.str.len().between(50, 10000)]
    df = df[~df.sequence.str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)]
    df = df.assign(sequence=df.sequence.str.slice(0, MAX_SEQ_LEN))
    return df.reset_index(drop=True)


def main():
    check_cli()
    rng = random.Random(SEED)

    defence = pd.read_parquet(INTERIM / "defence_proteins.parquet")
    print(f"{len(defence):,} defence proteins")

    if "assembly" not in defence.columns:
        sys.exit("no 'assembly' column — rerun fetch_annotations.py")

    assemblies = sorted(defence.assembly.dropna().unique())
    chosen = rng.sample(assemblies, min(N_GENOMES, len(assemblies)))
    print(f"sampling {len(chosen)} of {len(assemblies):,} genomes")

    zips = download_genomes(chosen)
    print(f"{len(zips)} genomes downloaded")

    exclude = set(defence.protein_id)
    # Accessions in the export carry versions; match on both forms.
    exclude |= {a.split(".")[0] for a in exclude}

    pool = []
    for z in tqdm(zips, desc="parsing"):
        for acc, seq in read_proteins(z):
            if acc in exclude or acc.split(".")[0] in exclude:
                continue
            pool.append((acc, seq, z.stem))
    print(f"{len(pool):,} candidate background proteins")

    if len(pool) < N_BACKGROUND:
        print(f"only {len(pool):,} available — raise N_GENOMES for more")
    rng.shuffle(pool)

    bg = pd.DataFrame(pool[:int(N_BACKGROUND * 1.1)],
                      columns=["protein_id", "sequence", "assembly"])
    bg = clean(bg).drop_duplicates(subset="protein_id").head(N_BACKGROUND)
    bg["system_type"] = pd.NA
    bg["label_binary"] = "background"
    print(f"{len(bg):,} background proteins after cleaning")

    defence = defence.copy()
    defence["label_binary"] = "defence"

    cols = ["protein_id", "sequence", "system_type", "assembly", "label_binary"]
    combined = pd.concat([defence.reindex(columns=cols),
                          bg.reindex(columns=cols)], ignore_index=True)
    combined = combined.drop_duplicates(subset="protein_id").reset_index(drop=True)

    print(f"\ncombined: {len(combined):,} proteins")
    print(combined.label_binary.value_counts().to_string())

    out = INTERIM / "dataset.parquet"
    combined.to_parquet(out, index=False)
    print(f"\nwrote {out}")
    print("\nNext: make_splits.py (re-clusters over the combined set), "
          "then embed.py (only the new sequences), then train_eval.py")


if __name__ == "__main__":
    main()
