"""One ESM-2 pass over every sequence, mean-pooled over residues, cached to disk.

Both tasks read this cache, so this is the only GPU-bound step. Sequences are
sorted by length and batched by token count rather than sequence count, which
keeps memory flat and avoids padding waste.

Resumable: if embeddings.npz exists, already-embedded ids are skipped.
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import EMBED_BATCH_TOKENS, ESM_LAYER, ESM_MODEL, INTERIM

OUT = INTERIM / "embeddings.npz"


def batches(records, max_tokens):
    records = sorted(records, key=lambda r: len(r[1]))
    batch, longest = [], 0
    for rec in records:
        longest = max(longest, len(rec[1]))
        if batch and longest * (len(batch) + 1) > max_tokens:
            yield batch
            batch, longest = [rec], len(rec[1])
        else:
            batch.append(rec)
    if batch:
        yield batch


def main():
    df = pd.read_parquet(INTERIM / "dataset_split.parquet")

    cached = {}
    if OUT.exists():
        z = np.load(OUT)
        cached = {k: z[k] for k in z.files}
        print(f"resuming, {len(cached):,} already embedded")

    todo = [
        (r.protein_id, r.sequence)
        for r in df.itertuples()
        if r.protein_id not in cached
    ]
    if not todo:
        print("nothing to do")
        return
    print(f"embedding {len(todo):,} sequences with {ESM_MODEL}")

    model, alphabet = torch.hub.load("facebookresearch/esm:main", ESM_MODEL)
    converter = alphabet.get_batch_converter()
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("no GPU found — this will be slow; use a smaller ESM model")
    model = model.to(device)

    done = 0
    for batch in tqdm(list(batches(todo, EMBED_BATCH_TOKENS))):
        _, _, tokens = converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[ESM_LAYER])
        reps = out["representations"][ESM_LAYER]

        for i, (pid, seq) in enumerate(batch):
            # Skip BOS at 0 and EOS at len+1; mean over real residues only.
            cached[pid] = reps[i, 1 : len(seq) + 1].mean(0).cpu().numpy()

        done += len(batch)
        if done % 2000 < len(batch):
            np.savez_compressed(OUT, **cached)

    np.savez_compressed(OUT, **cached)
    dim = next(iter(cached.values())).shape[0]
    print(f"wrote {OUT} — {len(cached):,} vectors of dim {dim}")


if __name__ == "__main__":
    main()
