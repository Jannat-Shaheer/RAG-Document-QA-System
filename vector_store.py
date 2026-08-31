"""Embedding + FAISS vector-store helpers.

Embeddings are L2-normalized and stored in a FAISS inner-product index, so a
search score is the cosine similarity between query and chunk (range -1..1,
higher = more similar).
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


def load_embedding_model():
    """Load the sentence-transformer used for both indexing and querying."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(model, sentences):
    """Encode a list of strings into float32, L2-normalized vectors."""
    vectors = model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def create_embeddings(model, chunks):
    """Return (embeddings, records) for a list of LangChain document chunks.

    ``records`` is a list of dicts:
        {"content": str, "page": int | None, "source": str}
    """
    records = [
        {
            "content": chunk.page_content,
            "page": chunk.metadata.get("page"),
            "source": Path(chunk.metadata.get("source", "?")).name,
        }
        for chunk in chunks
    ]
    embeddings = embed(model, [r["content"] for r in records])
    return embeddings, records


def store_in_faiss(embeddings):
    """Build an inner-product FAISS index (cosine similarity on normalized vecs)."""
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def extend_index(embeddings, new_records):
    """Append embeddings to the on-disk index, or create it if absent.

    Returns (index, all_records). Used for multi-document collections.
    """
    embeddings = np.asarray(embeddings, dtype="float32")
    if index_exists():
        index, records = load_index()
    else:
        index, records = faiss.IndexFlatIP(embeddings.shape[1]), []
    index.add(embeddings)
    records = records + list(new_records)
    save_index(index, records)
    return index, records


def save_index(index, records):
    config.ensure_dirs()
    faiss.write_index(index, str(config.INDEX_PATH))
    with open(config.TEXTS_PATH, "wb") as f:
        pickle.dump(records, f)


def index_exists() -> bool:
    return config.INDEX_PATH.exists() and config.TEXTS_PATH.exists()


def load_index():
    """Load (index, records) from disk. Raises FileNotFoundError if missing."""
    if not index_exists():
        raise FileNotFoundError(
            f"No index found at {config.INDEX_PATH}. Process a document first."
        )
    index = faiss.read_index(str(config.INDEX_PATH))
    with open(config.TEXTS_PATH, "rb") as f:
        records = pickle.load(f)
    return index, records
