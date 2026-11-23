"""Train every method on every task under both splits, write results/summary.csv.

Task A (family) uses only rows with a system_type, so background proteins added
for Task B are excluded from it rather than becoming a thirteenth class.

The output table is the deliverable: one row per (task, split, features, model),
so the random-vs-clustered gap can be read off directly.
"""

import itertools
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import INTERIM, RESULTS, SEED

AA = "ACDEFGHIKLMNPQRSTVWY"
# Murphy-style reduced alphabet: keeps the 3-mer space small.
REDUCED = {
    **dict.fromkeys("LVIM", "A"), **dict.fromkeys("C", "C"),
    **dict.fromkeys("AG", "G"), **dict.fromkeys("ST", "S"),
    **dict.fromkeys("P", "P"), **dict.fromkeys("FYW", "F"),
    **dict.fromkeys("EDNQ", "E"), **dict.fromkeys("KR", "K"),
    **dict.fromkeys("H", "H"),
}


def aac_features(seqs):
    idx = {a: i for i, a in enumerate(AA)}
    X = np.zeros((len(seqs), len(AA)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for a, n in Counter(s).items():
            if a in idx:
                X[i, idx[a]] = n / len(s)
    return X


def kmer_features(seqs, k=3):
    alpha = sorted(set(REDUCED.values()))
    space = {"".join(p): i for i, p in enumerate(itertools.product(alpha, repeat=k))}
    X = np.zeros((len(seqs), len(space)), dtype=np.float32)
    for i, s in enumerate(seqs):
        red = "".join(REDUCED.get(a, "G") for a in s)
        for j in range(len(red) - k + 1):
            key = red[j:j + k]
            if key in space:
                X[i, space[key]] += 1
        total = X[i].sum()
        if total:
            X[i] /= total
    return X


def evaluate(X, y, split, model_name):
    tr, te = split == "train", split == "test"
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                            early_stopping=True, random_state=SEED)
    clf.fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    return {
        "accuracy": accuracy_score(y[te], pred),
        "macro_f1": f1_score(y[te], pred, average="macro"),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
    }


def main():
    df = pd.read_parquet(INTERIM / "dataset_split.parquet")
    cache = dict(np.load(INTERIM / "embeddings.npz"))

    missing = (~df.protein_id.isin(cache)).sum()
    if missing:
        print(f"{missing:,} sequences have no embedding yet — run embed.py")
    df = df[df.protein_id.isin(cache)].reset_index(drop=True)
    print(f"{len(df):,} sequences with embeddings")

    # Each task names the rows it uses, so Task A ignores background proteins.
    tasks = {}
    has_type = df.system_type.notna().values
    if has_type.sum():
        tasks["A_family"] = (has_type, df.system_type.values)

    if "label_binary" in df.columns and df.label_binary.nunique() > 1:
        tasks["B_detection"] = (np.ones(len(df), bool), df.label_binary.values)
    else:
        print("only one class in label_binary — skipping Task B")

    rows = []
    for task, (mask, labels) in tasks.items():
        sub = df[mask]
        seqs = sub.sequence.tolist()
        feature_sets = {
            "esm2": np.vstack([cache[p] for p in sub.protein_id]),
            "aac": aac_features(seqs),
            "kmer": kmer_features(seqs),
        }
        y = labels[mask]
        print(f"\n{task}: {len(sub):,} rows, {len(set(y))} classes")

        for split_col in ("split_random", "split_clustered"):
            split = sub[split_col].values
            for feat_name, X in feature_sets.items():
                for model_name in ("logreg", "mlp"):
                    m = evaluate(X, y, split, model_name)
                    rows.append({"task": task,
                                 "split": split_col.replace("split_", ""),
                                 "features": feat_name, "model": model_name,
                                 **m})
                    print(f"  {split_col[6:]:10s} {feat_name:5s} "
                          f"{model_name:6s}  macro-F1 {m['macro_f1']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "summary.csv", index=False)

    pivot = out.pivot_table(index=["task", "features", "model"],
                            columns="split", values="macro_f1")
    pivot["leak"] = pivot["random"] - pivot["clustered"]
    pivot.round(3).to_csv(RESULTS / "leakage_gap.csv")

    print(f"\nwrote {RESULTS / 'summary.csv'}")
    print(f"\n{pivot.round(3).to_string()}")


if __name__ == "__main__":
    main()
