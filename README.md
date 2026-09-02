# RAG Document QA System

A Retrieval-Augmented Generation pipeline that answers questions about your own
PDF documents. Text is chunked and embedded with a sentence-transformer, indexed
in FAISS, and the top matches are passed as context to a local LLM (via Ollama)
that writes the answer with source citations.

## Architecture

```
PDF ──► ingest.py ──► vector_store.py ──► FAISS index (vectorstore/)
        load+chunk     embed (MiniLM,           │
                       L2-normalized)           ▼
                                        retriever.py ──► top-k chunks
                                        (cosine similarity)   │
                                                              ▼
                                        generator.py ──► answer
                                        (Ollama LLM)
```

| File | Responsibility |
|------|----------------|
| `config.py` | All tunable parameters (paths, chunk size, model names, `k`, threshold). Override via `RAG_*` env vars. |
| `ingest.py` | Load a PDF and split it into overlapping chunks. |
| `vector_store.py` | Embed chunks; build / load / add-to / remove-from the FAISS store (`index.faiss` + `embeddings.npy` + `texts.pkl`). |
| `retriever.py` | Embed the query and return the top-k chunks above the similarity threshold. |
| `generator.py` | Build a grounded prompt from the chunks and call the Ollama LLM. |
| `app.py` | Streamlit UI: upload, process (replace or append), list/remove indexed docs, ask, view sources. |
| `build_index.py` | Build the index from a PDF on the command line. |
| `evaluation.py` | End-to-end retrieval + answer metrics over the shared corpus, by document type (needs Ollama). |
| `benchmark.py` | Retrieval-only ablation: embedding model × chunk size, Recall@k / MRR / nDCG. See [`EVALUATION.md`](EVALUATION.md). |
| `eval/corpus.py` | Shared corpus definition + chunking + cached embeddings, used by both eval scripts. |
| `eval/build_corpus.py` | Fetch the 5-document, 4-type evaluation corpus. |

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Local LLM
# install Ollama from https://ollama.com, then:
ollama pull mistral        # default; `ollama pull llama3.2:3b` for lower latency
ollama serve
```

## Usage

**Web app**

```bash
streamlit run app.py
```

Upload a PDF, click *Process Document*, then ask questions. The answer streams in
token by token. Choose *Add to index* to keep earlier documents searchable in the
same collection. The **Indexed Documents** panel in the sidebar lists everything
currently searchable; click ✕ next to a document to drop its chunks from the
context (the index is rebuilt from the cached embeddings, no re-embedding).

> On CPU, `mistral` takes 1-3 min per answer. Set `RAG_LLM_MODEL=llama3.2:3b` for
> ~3x faster responses (at the cost of more "not in the document" refusals).

**Command line index build**

```bash
python build_index.py data/your_document.pdf
```

**Evaluation**

Both layers run over the **same shared corpus** — five public-domain documents of
four types (2 research papers, a government report, a technical book, classic
prose), defined in [`eval/corpus.py`](eval/corpus.py) with 25 labelled queries in
[`eval/queries.json`](eval/queries.json). Neither script is tied to a single book.

```bash
python eval/build_corpus.py     # once: fetch the corpus

# 1. Retrieval-only ablation (no LLM): embedding model x chunk size
python benchmark.py             # -> eval/results*.{csv,md}, eval/figures/
python benchmark.py --quick

# 2. End-to-end (needs Ollama): retrieval + answer keyword recall + LLM judge,
#    aggregated by document type
python evaluation.py --quick            # first 5 queries
python evaluation.py --judge            # all 25 (~2 min/query on CPU)
python evaluation.py --docs transformer.pdf,rag_paper.pdf
```

The retrieval ablation is written up in **[`EVALUATION.md`](EVALUATION.md)**.
Headline: chunk size matters more than embedding-model choice; a 384-dim model
(`bge-small-en-v1.5` @ 1536/192) tops the grid at Recall@5 = 0.88 / MRR@10 = 0.79;
retrieval is near-perfect on structured documents but preprocessing-bound on
heavily foot-noted prose (Recall@5 = 0.40).

Add your own documents to `CORPUS` in `eval/corpus.py` and queries to
`eval/queries.json`.

## Configuration

Defaults live in `config.py` and can be overridden with environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `RAG_CHUNK_SIZE` | 900 | characters per chunk |
| `RAG_CHUNK_OVERLAP` | 150 | overlap between chunks |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformer name |
| `RAG_TOP_K` | 6 | chunks retrieved per query |
| `RAG_SIMILARITY_THRESHOLD` | 0.25 | min cosine similarity to keep a chunk |
| `RAG_LLM_MODEL` | `mistral` | Ollama model for generation (`llama3.2:3b` = faster, more refusals) |

## Notes

- `data/` and `vectorstore/` are git-ignored; the index is rebuilt locally.
- Scanned (image-only) PDFs have no extractable text and are rejected at ingest.
