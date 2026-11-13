.PHONY: all clean
PY := python src

all: results/summary.csv

data/interim/annotations.parquet:
	$(PY)/fetch_annotations.py

data/interim/defence_proteins.parquet: data/interim/annotations.parquet
	$(PY)/build_dataset.py

data/interim/dataset_split.parquet: data/interim/defence_proteins.parquet
	$(PY)/make_splits.py

data/interim/embeddings.npz: data/interim/dataset_split.parquet
	$(PY)/embed.py

results/summary.csv: data/interim/embeddings.npz
	$(PY)/train_eval.py

clean:
	rm -rf data/interim results/*.csv
