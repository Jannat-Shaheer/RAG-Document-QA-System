"""Retrieval benchmark: embedding-model x chunk-size ablation.

Runs a grid of (embedding model, chunk size) over a fixed corpus of five
public-domain documents spanning four types, and reports Recall@k, MRR@10 and
nDCG@10 against a keyword-based ground truth (see eval/queries.json).

No LLM is involved - this measures the retrieval stage only, so the full grid
runs in minutes rather than hours.

Usage:
    python eval/build_corpus.py        # once, to fetch the documents
    python benchmark.py                # full grid
    python benchmark.py --quick        # 2 models x 2 chunk sizes, for a sanity check

Outputs (in eval/):
    results.csv              tidy long-form results
    results_overall.md       config x metric table
    results_by_doctype.md    Recall@5 / MRR broken down by document type
    figures/*.png            recall-vs-chunk, MRR-by-doctype, timing
"""

import argparse
import math
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from eval.corpus import embed_corpus, load_queries, missing_documents

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"
FIGDIR = EVAL / "figures"

# --- grid -----------------------------------------------------------------
# model id -> query prefix (bge-v1.5 wants an instruction on the query side only)
MODELS = {
    "all-MiniLM-L6-v2": "",
    "all-mpnet-base-v2": "",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
CHUNK_CONFIGS = [(256, 32), (512, 64), (1024, 128), (1536, 192)]

QUICK_MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
QUICK_CHUNKS = [(256, 32), (1024, 128)]

K_VALUES = [1, 3, 5, 10]
TOP_K = 10
REL_MIN_KEYWORDS = 2  # a chunk is relevant if it contains >= this many keywords


# --- embedding + index --------------------------------------------------------

_MODEL_CACHE = {}


def get_model(model_id):
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = SentenceTransformer(model_id)
    return _MODEL_CACHE[model_id]


def encode(model, texts, batch_size=64):
    v = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return np.asarray(v, dtype="float32")


def build_index(model_id, chunk_size, overlap):
    """Chunk + embed the whole corpus for one grid cell (cached on disk)."""
    emb, records, index_time = embed_corpus(
        lambda texts: encode(get_model(model_id), texts),
        model_id, chunk_size, overlap,
    )
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, records, index_time


# --- metrics --------------------------------------------------------------

def relevant_flags(records, hit_indices, keywords):
    kws = [k.lower() for k in keywords]
    flags = []
    for idx in hit_indices:
        text = records[idx]["content"].lower()
        n = sum(1 for k in kws if k in text)
        flags.append(n >= min(REL_MIN_KEYWORDS, len(kws)))
    return flags


def ndcg_at_k(flags, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(flags[:k]) if rel)
    n_rel = sum(flags[:k])
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg else 0.0


def eval_query(model, index, records, q):
    qtext = q["_prefix"] + q["query"]
    t0 = time.time()
    qv = encode(model, [qtext])           # query embedding dominates this cost
    _, idx = index.search(qv, TOP_K)
    latency = time.time() - t0
    flags = relevant_flags(records, idx[0], q["keywords"])

    first = next((i for i, r in enumerate(flags) if r), None)
    row = {f"recall@{k}": float(any(flags[:k])) for k in K_VALUES}
    row["mrr@10"] = 1.0 / (first + 1) if first is not None else 0.0
    row["ndcg@10"] = ndcg_at_k(flags, TOP_K)
    row["latency_ms"] = latency * 1000
    return row


# --- driver -------------------------------------------------------------------

def run(models, chunk_configs, queries):
    rows = []
    for model_id in models:
        prefix = MODELS[model_id]
        for q in queries:
            q["_prefix"] = prefix
        for cs, co in chunk_configs:
            print(f"  {model_id:26s} chunk={cs:5d}/{co:<3d} ", end="", flush=True)
            index, records, index_time = build_index(model_id, cs, co)
            model = get_model(model_id)
            per_q = [eval_query(model, index, records, q) for q in queries]
            df = pd.DataFrame(per_q)
            df["doc_type"] = [q["doc_type"] for q in queries]
            agg = df.drop(columns="doc_type").mean()
            print(
                f"n_chunks={len(records):6d}  "
                f"R@5={agg['recall@5']:.2f}  MRR={agg['mrr@10']:.2f}  "
                f"nDCG={agg['ndcg@10']:.2f}"
            )
            for _, r in df.iterrows():
                rows.append(
                    {
                        "model": model_id,
                        "chunk_size": cs,
                        "overlap": co,
                        "n_chunks": len(records),
                        "index_time_s": index_time,
                        "doc_type": r["doc_type"],
                        **{m: r[m] for m in
                           [*[f"recall@{k}" for k in K_VALUES],
                            "mrr@10", "ndcg@10", "latency_ms"]},
                    }
                )
    return pd.DataFrame(rows)


def _md_table(frame, index=False):
    """Minimal GitHub-markdown table (avoids a tabulate dependency)."""
    f = frame.reset_index() if index else frame
    cols = list(f.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in f.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_tables(df):
    metrics = [*[f"recall@{k}" for k in K_VALUES], "mrr@10", "ndcg@10", "latency_ms"]
    overall = (
        df.groupby(["model", "chunk_size", "overlap"])[metrics + ["n_chunks"]]
        .mean()
        .round(3)
        .reset_index()
        .sort_values(["model", "chunk_size"])
    )
    EVAL.joinpath("results.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    EVAL.joinpath("results_overall.md").write_text(
        f"# Retrieval benchmark - overall (mean over {df.groupby(['model','chunk_size']).size().iloc[0]} queries per cell)\n\n"
        + _md_table(overall),
        encoding="utf-8",
    )

    best = overall.sort_values("recall@5", ascending=False).iloc[0]
    mask = (
        (df.model == best.model)
        & (df.chunk_size == best.chunk_size)
        & (df.overlap == best.overlap)
    )
    by_type = (
        df[mask].groupby("doc_type")[["recall@1", "recall@5", "mrr@10", "ndcg@10"]]
        .mean()
        .round(3)
    )
    EVAL.joinpath("results_by_doctype.md").write_text(
        f"# Best config: {best.model}, chunk {int(best.chunk_size)}/"
        f"{int(best.overlap)}\n\nRecall@5 = {best['recall@5']:.3f}, "
        f"MRR@10 = {best['mrr@10']:.3f}\n\n## By document type\n\n"
        + _md_table(by_type, index=True),
        encoding="utf-8",
    )
    return overall, best


def make_figures(df, overall, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)

    # 1. Recall@5 vs chunk size, one line per model
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_id, g in overall.groupby("model"):
        g = g.sort_values("chunk_size")
        ax.plot(g.chunk_size, g["recall@5"], marker="o", label=model_id)
    ax.set_xlabel("chunk size (characters)")
    ax.set_ylabel("Recall@5")
    ax.set_title("Retrieval Recall@5 vs chunk size")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "recall_vs_chunksize.png", dpi=150)
    plt.close(fig)

    # 2. MRR@10 by document type, best config
    mask = (
        (df.model == best.model)
        & (df.chunk_size == best.chunk_size)
        & (df.overlap == best.overlap)
    )
    by_type = df[mask].groupby("doc_type")["mrr@10"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(by_type.index, by_type.values, color="#4C72B0")
    ax.set_xlabel("MRR@10")
    ax.set_xlim(0, 1)
    ax.set_title(f"MRR@10 by document type\n({best.model}, {int(best.chunk_size)} chars)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "mrr_by_doctype.png", dpi=150)
    plt.close(fig)

    # 3. indexing time & mean query latency per model (at 512/64)
    ref = df[df.chunk_size == 512].groupby("model").agg(
        index_time_s=("index_time_s", "mean"), latency_ms=("latency_ms", "mean")
    )
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.bar(ref.index, ref.index_time_s, color="#55A868")
    a1.set_ylabel("seconds")
    a1.set_title("Corpus embedding time (512/64)")
    a1.tick_params(axis="x", rotation=30, labelsize=7)
    a2.bar(ref.index, ref.latency_ms, color="#C44E52")
    a2.set_ylabel("ms")
    a2.set_title("Mean query latency (embed + search)")
    a2.tick_params(axis="x", rotation=30, labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "timing.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="2 models x 2 chunk sizes")
    args = parser.parse_args()

    missing = missing_documents()
    if missing:
        sys.exit(f"Missing documents: {missing}\nRun: python eval/build_corpus.py")

    queries = load_queries()
    models = QUICK_MODELS if args.quick else list(MODELS)
    chunks = QUICK_CHUNKS if args.quick else CHUNK_CONFIGS

    print(f"Grid: {len(models)} models x {len(chunks)} chunk configs, "
          f"{len(queries)} queries\n" + "=" * 70)
    t0 = time.time()
    df = run(models, chunks, queries)
    overall, best = write_tables(df)
    make_figures(df, overall, best)

    print("=" * 70)
    print(overall.to_string(index=False))
    print(f"\nBest by Recall@5: {best.model}, chunk {int(best.chunk_size)}/"
          f"{int(best.overlap)}  (R@5={best['recall@5']:.3f})")
    print(f"\nWrote eval/results.csv, eval/results_overall.md, "
          f"eval/results_by_doctype.md, eval/figures/*.png")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
