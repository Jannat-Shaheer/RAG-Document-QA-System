"""Embedding + FAISS vector-store helpers.

Embeddings are L2-normalized and stored in a FAISS inner-product index, so a
search score is the cosine similarity between query and chunk (range -1..1,
higher = more similar).

Three files make up the store (all under ``vectorstore/``):
    index.faiss    - the FAISS index, for fast query-time loading
    embeddings.npy - the raw vectors, kept so a document can be dropped and the
                     index rebuilt without re-embedding everything
    texts.pkl      - the chunk records: {"content", "page", "source"}
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


# --- persistence ---------------------------------------------------------------

def index_exists() -> bool:
    """True if the query-time files (index + records) are present."""
    return config.INDEX_PATH.exists() and config.TEXTS_PATH.exists()


def _build_index(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_store(embeddings, records):
    """Write all three store files, rebuilding the FAISS index from scratch."""
    config.ensure_dirs()
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.write_index(_build_index(embeddings), str(config.INDEX_PATH))
    np.save(config.EMB_PATH, embeddings)
    with open(config.TEXTS_PATH, "wb") as f:
        pickle.dump(records, f)


def clear_store():
    for path in (config.INDEX_PATH, config.EMB_PATH, config.TEXTS_PATH):
        path.unlink(missing_ok=True)


def load_index():
    """Load (index, records) for querying. Raises FileNotFoundError if missing."""
    if not index_exists():
        raise FileNotFoundError(
            f"No index found at {config.INDEX_PATH}. Process a document first."
        )
    index = faiss.read_index(str(config.INDEX_PATH))
    with open(config.TEXTS_PATH, "rb") as f:
        records = pickle.load(f)
    return index, records


def load_records():
    """Load just the chunk records (no embeddings / index)."""
    if not config.TEXTS_PATH.exists():
        return []
    with open(config.TEXTS_PATH, "rb") as f:
        return pickle.load(f)


def _load_embeddings():
    if not config.EMB_PATH.exists():
        raise FileNotFoundError(
            "Embeddings cache missing - re-process the documents to enable "
            "per-document removal."
        )
    return np.load(config.EMB_PATH)


# --- document-level operations -----------------------------------------------

def add_to_store(new_embeddings, new_records):
    """Append a document's chunks to the store. Returns the full record list."""
    new_embeddings = np.asarray(new_embeddings, dtype="float32")
    if index_exists():
        embeddings = np.vstack([_load_embeddings(), new_embeddings])
        records = load_records() + list(new_records)
    else:
        embeddings, records = new_embeddings, list(new_records)
    save_store(embeddings, records)
    return records


def list_documents():
    """Return [(source, chunk_count), ...] for everything currently indexed."""
    counts = {}
    for rec in load_records():
        src = rec.get("source", "?")
        counts[src] = counts.get(src, 0) + 1
    return sorted(counts.items())


def remove_document(source):
    """Drop every chunk from ``source`` and rebuild the index.

    Returns the remaining records (empty list if the store is now empty).
    """
    embeddings = _load_embeddings()
    records = load_records()
    keep = [i for i, r in enumerate(records) if r.get("source") != source]

    if not keep:
        clear_store()
        return []

    save_store(embeddings[keep], [records[i] for i in keep])
    return [records[i] for i in keep]
