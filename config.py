"""Central configuration for the RAG document-QA system.

Every tunable parameter lives here so the ingest / embedding / retrieval /
generation stages stay in sync. Values can be overridden with environment
variables (useful for the evaluation script or a deployment).
"""

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
EMB_PATH = VECTORSTORE_DIR / "embeddings.npy"
TEXTS_PATH = VECTORSTORE_DIR / "texts.pkl"
UPLOAD_PATH = DATA_DIR / "temp.pdf"

# --- Chunking ------------------------------------------------------------------
# Larger chunks keep enumerated lists / tables intact so retrieval can return a
# whole list in one hit.
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 150))

# --- Embeddings --------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Retrieval -------------------------------------------------------------
TOP_K = int(os.getenv("RAG_TOP_K", 6))
# Minimum cosine similarity for a chunk to count as relevant (range -1..1).
# Embeddings are L2-normalized, so FAISS inner product == cosine similarity.
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", 0.25))

# --- Generation --------------------------------------------------------------
# mistral (7B) synthesizes answers from multi-chunk context far more reliably
# than 3B models on this data. Override with RAG_LLM_MODEL=llama3.2:3b for
# lower CPU latency at the cost of more "not in the document" refusals.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "mistral")

# Sentinel returned by the retriever when nothing clears the threshold.
NO_CONTEXT_SENTINEL = "__NO_RELEVANT_CONTEXT__"


def ensure_dirs() -> None:
    """Create the data/ and vectorstore/ directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    VECTORSTORE_DIR.mkdir(exist_ok=True)
