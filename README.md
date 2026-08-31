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
| `vector_store.py` | Embed chunks, build / save / load / extend the FAISS index. |
| `retriever.py` | Embed the query and return the top-k chunks above the similarity threshold. |
| `generator.py` | Build a grounded prompt from the chunks and call the Ollama LLM. |
| `app.py` | Streamlit UI: upload, process (replace or append), ask, view sources. |
| `build_index.py` | Build the index from a PDF on the command line. |
| `evaluation.py` | Offline retrieval + answer metrics over a set of test questions. |

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
same collection.

> On CPU, `mistral` takes 1-3 min per answer. Set `RAG_LLM_MODEL=llama3.2:3b` for
> ~3x faster responses (at the cost of more "not in the document" refusals).

**Command line index build**

```bash
python build_index.py data/your_document.pdf
```

**Evaluation**

```bash
python evaluation.py            # retrieval + answer keyword metrics
python evaluation.py --judge    # also grade answers with an LLM judge
```

Reports:
- *Retrieval hit@k* – a relevant keyword appeared in the retrieved chunks
- *Retrieval keyword recall* – fraction of expected keywords retrieved
- *Answer keyword recall* – fraction of expected keywords in the final answer
- *Avg latency* – seconds per query (retrieval + generation)

Edit `TEST_CASES` in `evaluation.py` to match your document.

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
