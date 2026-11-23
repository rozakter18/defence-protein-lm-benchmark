from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RESULTS = ROOT / "results"

for d in (RAW, INTERIM, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# NCBI Entrez. Set your own address: NCBI blocks anonymous bulk requests.
# An API key raises the rate limit from 3 to 10 requests/second.
ENTREZ_EMAIL = "rozakterofficial18@gmail.com"
ENTREZ_API_KEY = None

# How many defence system types to keep. Rarest classes are dropped so every
# class has enough sequences to split and train on.
N_SYSTEM_TYPES = 12
MIN_SEQS_PER_TYPE = 200

# Sequences longer than this are truncated before embedding. ESM-2 memory grows
# quadratically with length; 1022 is the model's own positional limit.
MAX_SEQ_LEN = 1022

# MMseqs2 clustering identity for the leakage-controlled split.
CLUSTER_IDENTITY = 0.30
CLUSTER_COVERAGE = 0.80

TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
SEED = 0

# esm2_t12_35M_UR50D   — fastest, fits anywhere
# esm2_t30_150M_UR50D  — good default on a free T4
# esm2_t33_650M_UR50D  — needs batching care, slow on free tier
ESM_MODEL = "esm2_t30_150M_UR50D"
ESM_LAYER = 30
EMBED_BATCH_TOKENS = 8192

# Cap sequences per system type. Balances classes and keeps the CPU
# embedding pass to about an hour. Set to None to use everything.
MAX_SEQS_PER_TYPE = 2000

# Cap sequences per system type. Balances classes and keeps the CPU
# embedding pass to about an hour. Set to None to use everything.
MAX_SEQS_PER_TYPE = 2000
