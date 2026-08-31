"""Semantic retrieval over the FAISS index."""

import config
from vector_store import embed


def search(query, model, index, records, k=None, threshold=None):
    """Return (chunks, scores) for the top matches above the similarity threshold.

    ``chunks`` is a list of dicts {"content": str, "page": int | None}.
    ``scores`` is the matching list of cosine similarities (higher = better).
    Both are empty when nothing clears the threshold.
    """
    k = k or config.TOP_K
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold

    query_vector = embed(model, [query])
    similarities, indices = index.search(query_vector, k)

    chunks, scores = [], []
    for score, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        if score >= threshold:
            chunks.append(records[idx])
            scores.append(float(score))

    return chunks, scores
