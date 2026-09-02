"""Shared evaluation corpus.

Five public-domain documents spanning four types. Used by both `benchmark.py`
(retrieval ablation) and `evaluation.py` (end-to-end), so neither is tied to a
single document. Build it once with `python eval/build_corpus.py`.
"""

import json
import time
from pathlib import Path

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "eval" / "cache"

# name -> (path, doc_type, max_pages or None)
CORPUS = {
    "transformer.pdf": (DATA / "eval" / "transformer.pdf", "research paper", None),
    "rag_paper.pdf": (DATA / "eval" / "rag_paper.pdf", "research paper", None),
    "nist_ai_rmf.pdf": (DATA / "eval" / "nist_ai_rmf.pdf", "government report", None),
    "art_of_war.pdf": (DATA / "eval" / "art_of_war.pdf", "classic prose", None),
    # cap the 796-page book to the part the queries target, to keep it
    # comparable in size to the other documents
    "EffectiveProjectManagement_Wysocki.pdf": (
        DATA / "EffectiveProjectManagement_Wysocki.pdf", "technical book", 250,
    ),
}

QUERIES_PATH = ROOT / "eval" / "queries.json"

# Some retrieval models expect an instruction on the query side only.
_QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def query_prefix(model_id):
    return _QUERY_PREFIX.get(model_id, "")


def missing_documents():
    return [name for name, (path, _, _) in CORPUS.items() if not path.exists()]


def load_queries():
    return json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]


def load_pages(name):
    path, doc_type, max_pages = CORPUS[name]
    pages = PyPDFLoader(str(path)).load()
    if max_pages:
        pages = pages[:max_pages]
    for p in pages:
        p.metadata["source"] = name
        p.metadata["doc_type"] = doc_type
    return pages


def chunk_corpus(chunk_size, overlap, only=None):
    """List of chunk dicts {content, source, doc_type} for the whole corpus.

    ``only`` optionally restricts to a set/list of document names.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    names = [n for n in CORPUS if only is None or n in only]
    records = []
    for name in names:
        for ch in splitter.split_documents(load_pages(name)):
            records.append(
                {
                    "content": ch.page_content,
                    "source": name,
                    "doc_type": ch.metadata["doc_type"],
                }
            )
    return records


def embed_corpus(encode_fn, model_id, chunk_size, overlap):
    """Chunk + embed the whole corpus, cached on disk under eval/cache/.

    ``encode_fn(list[str]) -> float32[n, d]`` does the actual embedding, so the
    caller controls the model. Returns (embeddings, records, embed_seconds).
    Both benchmark.py and evaluation.py share this cache.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{model_id.replace('/', '_')}__{chunk_size}_{overlap}"
    npz, meta, tim = (CACHE / f"{tag}.{ext}" for ext in ("npz", "json", "time"))

    if npz.exists() and meta.exists():
        emb = np.load(npz)["emb"]
        records = json.loads(meta.read_text(encoding="utf-8"))
        seconds = float(tim.read_text()) if tim.exists() else float("nan")
        return emb, records, seconds

    records = chunk_corpus(chunk_size, overlap)
    t0 = time.time()
    emb = np.asarray(encode_fn([r["content"] for r in records]), dtype="float32")
    seconds = time.time() - t0
    np.savez_compressed(npz, emb=emb)
    meta.write_text(json.dumps(records), encoding="utf-8")
    tim.write_text(str(seconds))
    return emb, records, seconds
