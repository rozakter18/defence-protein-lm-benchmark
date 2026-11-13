# Benchmarking protein language model embeddings on bacterial antiphage defence proteins

Do ESM-2 embeddings classify bacterial defence system proteins better than the
profile HMMs and simple sequence features already used for this — and how much
of any apparent advantage disappears when homologous sequences are kept out of
the test set?

## Why the split design is the point

Defence system proteins come in homologous families. A random train/test split
puts near-identical sequences on both sides, so a classifier can score well by
recognising sequences it has effectively already seen. Every number here is
therefore reported twice:

- **random split** — the naive baseline, included to show the size of the leak
- **clustered split** — sequences clustered by MMseqs2 at 30% identity, whole
  clusters assigned to one side only

The gap between the two is the headline result.

## Tasks

Both tasks reuse the same cached embeddings, so the GPU pass runs once.

| Task | Question | Type |
|---|---|---|
| A — family | Which defence system type is this protein from? | multi-class |
| B — detection | Is this a defence protein or a background protein? | binary |

## Methods compared

| Method | What it is |
|---|---|
| `esm2` | mean-pooled ESM-2 residue embeddings + classifier |
| `aac` | amino-acid composition (20 features) |
| `kmer` | 3-mer composition, reduced alphabet |
| `hmm` | profile HMM bit-scores — the incumbent method |

Classifiers: logistic regression and a small MLP, on frozen features.

## Running it

```bash
pip install -r requirements.txt
make all
```

Or step by step:

```bash
python src/fetch_annotations.py     # DefenseFinder precomputed RefSeq tables
python src/build_dataset.py         # join annotations to protein sequences
python src/make_splits.py           # MMseqs2 clustering, both split types
python src/embed.py                 # ESM-2 pass, cached to disk
python src/train_eval.py            # all methods, both tasks, both splits
```

`mmseqs` must be on PATH. On Colab: `!conda install -c bioconda mmseqs2 -y`
or download the static binary from the MMseqs2 releases page.

## Results

Filled in after the first full run. Table lives in `results/summary.csv`.

## Limitations

- Defence system labels come from DefenseFinder's HMM-based annotation, not
  experimental validation. The labels are predictions, so this benchmarks
  agreement with an existing tool rather than ground truth.
- Comparing against `hmm` scores is partly circular for the same reason: the
  labels were produced by those HMMs. The comparison is still informative about
  whether embeddings recover the same signal from sequence alone, but it cannot
  show embeddings beating HMMs at the task that defined the labels.
- 30% identity is a conventional threshold, not a principled one. Results at
  other thresholds are not explored here.
- Background proteins for Task B are sampled from the same genomes and are
  assumed non-defence because DefenseFinder did not flag them. Some are
  probably unrecognised defence proteins.

## Layout

```
src/        pipeline scripts, run in order
data/       downloaded and derived data (gitignored)
results/    metrics tables and figures
```
