"""Build the FAISS index from a PDF, offline.

Usage:
    python build_index.py data/EffectiveProjectManagement_Wysocki.pdf
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
from ingest import load_pdf, split_text
from vector_store import create_embeddings, load_embedding_model, save_store


def build(pdf_path):
    print(f"Loading {pdf_path} ...")
    chunks = split_text(load_pdf(pdf_path))
    print(f"  {len(chunks)} chunks")

    print("Embedding ...")
    model = load_embedding_model()
    embeddings, records = create_embeddings(model, chunks)

    print("Building FAISS index ...")
    save_store(embeddings, records)

    print(f"Saved index ({len(records)} vectors) to {config.INDEX_PATH}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        config.DATA_DIR / "EffectiveProjectManagement_Wysocki.pdf"
    )
    build(path)
