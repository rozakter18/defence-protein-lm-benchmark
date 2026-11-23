# Homology leakage in protein language model benchmarks

**Do ESM-2 embeddings classify bacterial antiphage defence proteins because they
understand the proteins, or because they recognise sequences they have already
seen?**

Short answer: mostly the former — and the cheap baseline it is usually compared
against is the one doing the recognising.

Removing homologs from the test set costs ESM-2 embeddings **0.027 ± 0.010**
macro-F1. It costs 3-mer composition **0.242 ± 0.030** — nearly ten times more.
Most of what the k-mer baseline appeared to know was sequence-similarity
leakage.

![Macro-F1 under random and clustered splits for three feature sets](results/leakage_gap.png)

## Why this needs checking

Protein families are, by definition, sets of similar sequences. Split a protein
dataset at random and near-identical sequences land on both sides of the
train/test boundary, so a model can score well by matching against something it
has already memorised. The published number then measures recall of the training
set rather than generalisation.

The standard correction is to cluster by sequence identity and assign whole
clusters to one side. This repo does that, and reports both splits side by side
so the size of the correction is visible rather than assumed.

## Result

Macro-F1 on held-out test data, logistic regression on frozen features, mean ±
s.d. over 10 cluster-assignment seeds, clustering at 20% identity:

| Features | Random split | Clustered split | Leakage gap |
|---|---|---|---|
| ESM-2 150M embeddings | 0.987 ± 0.001 | 0.961 ± 0.009 | **0.027 ± 0.010** |
| 3-mer composition | 0.733 ± 0.006 | 0.490 ± 0.030 | **0.242 ± 0.030** |
| Amino-acid composition | 0.467 ± 0.005 | 0.389 ± 0.016 | 0.078 ± 0.016 |

The two gaps are separated by roughly seven standard deviations.

At the looser 30% identity threshold the picture is the same — ESM-2 gap 0.033,
3-mer gap 0.260 on a single split — so the conclusion does not depend on where
the threshold is drawn.

### A second finding, about precision rather than accuracy

Random splits do not only inflate the score. They also make it look more precise
than it is. Standard deviations across seeds:

| Features | Random split | Clustered split |
|---|---|---|
| ESM-2 | 0.001 | 0.009 |
| 3-mer | 0.006 | 0.030 |
| Amino-acid | 0.005 | 0.016 |

Three to five times tighter under random splitting. A single random-split number
would look reproducible to three decimal places while being wrong in the first.

### Where the errors fall

The clustered-split confusion matrix (30% identity, logistic regression, overall
accuracy 0.948) is not uniform, and the structure is informative.

Three classes are essentially solved: MazEF (F1 1.000), PD-T7-2 (1.000),
AbiE (0.998).

Two are notably harder:

- **Retron** (recall 0.759) loses 37 proteins to Cas and 32 to RM.
- **BREX** (recall 0.842) loses 28 to Lamassu-Fam, 17 to RM, 16 to Gabija.

These are not random failures. They fall between systems that share protein
domain families — BREX and RM both carry methyltransferases; retrons are built
around a reverse transcriptase, and CRISPR-associated reverse transcriptases
exist. The model is confusing systems that are genuinely related at the domain
level, which is closer to correct behaviour being scored as error than it is to
a model guessing.

This matters for interpreting the headline number. If 0.961 came from residual
leakage or a label artifact, errors would be spread randomly or concentrated in
one class. Instead they track known protein homology.

## What this does not show

**The prediction going in was the opposite.** The expectation was that
embeddings, encoding family identity, would be the most exposed to homology
leakage, and that simple composition features would degrade least. That was
wrong in both direction and magnitude. It is recorded here because a hypothesis
that failed is more informative than one never written down.

**It is not evidence that ESM-2 beats profile HMMs.** No HMM baseline is
included, deliberately: the class labels in this dataset were produced by
DefenseFinder's HMMs, so scoring HMMs against them would be close to circular.
The comparison here is between feature representations under identical
conditions, not a claim about the best available annotation tool.

## Data

Labels come from the [DefenseFinder](https://defensefinder.mdmlab.fr/) RefSeq
precomputed database — DefenseFinder run over the complete bacterial RefSeq
genomes, snapshot May 2022. The export contains 152,386 defence systems; after
deduplication that is 198,358 unique proteins.

Sequences are fetched from NCBI by accession. Of 23,789 requested, 23,770
resolved — the rest were suppressed or replaced since the 2022 snapshot.

The 12 most abundant system types are kept and **capped at 2,000 proteins each**
(PD-T7-2 has only 1,789). The cap is deliberate: the raw distribution runs from
64,481 RM proteins to 1,789 PD-T7-2, a 36:1 ratio under which macro-F1 is
dominated by sampling noise in the small classes. It also keeps the embedding
pass tractable on CPU.

Final dataset: **23,770 proteins across 12 defence system types.**

## Method

**Embeddings.** `esm2_t30_150M_UR50D`, final-layer residue representations
mean-pooled over the sequence, 640 dimensions. Computed once and cached;
sequences truncated to 1,022 residues.

**Clustering.** MMseqs2 `easy-cluster`, 80% coverage. At 20% identity,
sensitivity is raised to `-s 7.5`; MMseqs2's default prefilter does not reliably
find matches that remote, and without the increase a nominal 20% run returns
almost the same clusters as 30%.

| Identity | Clusters | Largest | Median |
|---|---|---|---|
| 30% | 3,559 | — | — |
| 20% | 2,163 | 537 | 2 |

That distribution — a few very large conserved families alongside a long tail of
pairs — is itself why repeated seeds are necessary. A 537-member cluster is 2.3%
of the dataset moving between train and test as a single block.

**Splitting.** Whole clusters are assigned to one side only. Assignment is
stratified by system type: clusters are grouped by their majority type and
filled per class to the target proportions. A naive largest-first assignment was
tried first and left four of twelve classes with **zero** test examples, which
makes macro-F1 meaningless — the stratification is load-bearing, not cosmetic.

**Evaluation.** Logistic regression and a small MLP on standardised frozen
features. 70/10/20 train/val/test. Macro-F1, because the question concerns all
twelve classes equally.

## Reproducing

```bash
conda create -n dplm python=3.11 -y && conda activate dplm
conda install -c bioconda -c conda-forge mmseqs2 -y
pip install -r requirements.txt
```

Download the RefSeq export from
https://defensefinder.mdmlab.fr/wiki/refseq/ into `data/raw/`, set
`ENTREZ_EMAIL` in `src/config.py`, then:

```bash
python src/fetch_annotations.py   # parse labels, cap per class
python src/build_dataset.py       # fetch sequences from NCBI (~5 min)
python src/make_splits.py         # MMseqs2 clustering, both splits
python src/embed.py               # ESM-2 pass — ~11 h CPU, minutes on GPU
python src/train_eval.py          # single-split results
python src/repeat_eval.py         # 10-seed means and spreads
python src/make_figures.py
```

`embed.py` caches every 2,000 sequences and skips what is already done, so it is
safe to interrupt and resume.

## Limitations

- **Labels are predictions, not ground truth.** DefenseFinder assigns them by
  HMM search. Mislabelled proteins propagate silently, and anything DefenseFinder
  cannot recognise is absent from the dataset entirely.
- **Clustering at 20% identity is near MMseqs2's practical limit.** It is a
  robustness check on the 30% result, not a stronger ground truth.
- **One model, one pooling strategy.** Mean-pooling over residues is the common
  default but discards positional structure; other pooling choices, other model
  sizes, and fine-tuning are untested here.
- **Twelve classes, subsampled.** Results may not extend to rare system types or
  to the full 152-type vocabulary.
- **Clusters are assigned by majority type.** A cluster spanning several system
  types is balanced once rather than for each, which slightly loosens per-class
  proportions.
- **Task B is not implemented.** A defence-versus-background detection task was
  designed (`build_dataset.py` describes the background sampling) but the
  background set was never built. Only family classification is reported.

## Layout

```
src/        pipeline scripts, run in order
data/       downloaded and derived data (gitignored)
results/    metrics tables and figures
```

`results/repeats_id0.2.csv` holds the per-seed numbers behind every figure in
this README. `results/id30/` holds the 30%-identity run.
