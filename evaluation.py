"""End-to-end evaluation of the RAG pipeline over the shared evaluation corpus.

Runs retrieval + generation for every query in eval/queries.json (25 questions
across five documents of four types) and measures both stages:

  * Retrieval -- fraction of the query's expected keywords present in the
    retrieved chunks (retrieval keyword recall).
  * Answer    -- fraction of expected keywords present in the generated answer,
    plus an optional 1-5 LLM-as-judge score (--judge).

Results are aggregated overall and broken down by document type, so the metrics
are not tied to any single book. Retrieval settings come from config.py
(RAG_EMBEDDING_MODEL, RAG_CHUNK_SIZE, RAG_TOP_K, ...).

Usage:
    python eval/build_corpus.py       # once, to fetch the documents
    python evaluation.py              # all 25 queries  (slow: ~2 min/query on CPU)
    python evaluation.py --quick      # first 5 queries
    python evaluation.py --limit 8 --judge
    python evaluation.py --docs transformer.pdf,rag_paper.pdf
"""

import argparse
import sys
import time
from collections import defaultdict

# PDF text often contains ligatures/smart quotes; force UTF-8 stdout on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import faiss
import numpy as np

import config
from eval.corpus import embed_corpus, load_queries, missing_documents, query_prefix
from generator import generate_answer
from vector_store import embed, load_embedding_model


def keyword_recall(text, keywords):
    """Fraction of keywords that appear (case-insensitive) in text."""
    low = text.lower()
    hits = [kw for kw in keywords if kw.lower() in low]
    return len(hits) / len(keywords), hits


def judge_answer(query, answer):
    """Ask the LLM to grade an answer 1-5. Returns int or None on failure."""
    import ollama

    prompt = (
        "Grade the answer to the question on a 1-5 scale "
        "(1 = wrong/irrelevant, 5 = fully correct and complete). "
        "Reply with only the digit.\n\n"
        f"Question: {query}\nAnswer: {answer}\nGrade:"
    )
    try:
        resp = ollama.chat(model=config.LLM_MODEL,
                           messages=[{"role": "user", "content": prompt}])
    except Exception as e:  # noqa: BLE001 - evaluation should not crash
        print(f"  (judge unavailable: {e})")
        return None
    for ch in resp["message"]["content"]:
        if ch in "12345":
            return int(ch)
    return None


def retrieve(model, index, records, query, prefix):
    """Mirror retriever.search over the in-memory eval index."""
    qv = embed(model, [prefix + query])
    sims, idx = index.search(qv, config.TOP_K)
    chunks, scores = [], []
    for score, i in zip(sims[0], idx[0]):
        if i != -1 and score >= config.SIMILARITY_THRESHOLD:
            chunks.append(records[i])
            scores.append(float(score))
    return chunks, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true",
                        help="also grade answers with an LLM judge")
    parser.add_argument("--quick", action="store_true",
                        help="first 5 queries only")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N queries")
    parser.add_argument("--docs", default=None,
                        help="comma-separated document filenames to include")
    args = parser.parse_args()

    missing = missing_documents()
    if missing:
        sys.exit(f"Missing documents: {missing}\nRun: python eval/build_corpus.py")

    queries = load_queries()
    if args.docs:
        wanted = {d.strip() for d in args.docs.split(",")}
        queries = [q for q in queries if q["doc"] in wanted]
    limit = 5 if args.quick else args.limit
    if limit:
        queries = queries[:limit]
    if not queries:
        sys.exit("No queries selected.")

    model_id = config.EMBEDDING_MODEL
    prefix = query_prefix(model_id)
    print(f"Embedding corpus with {model_id} at "
          f"chunk {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} ...")
    model = load_embedding_model()
    emb, records, embed_s = embed_corpus(
        lambda texts: embed(model, texts),
        model_id, config.CHUNK_SIZE, config.CHUNK_OVERLAP,
    )
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb, dtype="float32"))
    print(f"  {len(records)} chunks ({embed_s:.0f}s to embed, cached after)\n"
          f"Running {len(queries)} queries "
          f"(k={config.TOP_K}, threshold={config.SIMILARITY_THRESHOLD}, "
          f"model={config.LLM_MODEL})\n" + "=" * 72)

    by_type = defaultdict(lambda: defaultdict(list))
    rows = []

    for i, q in enumerate(queries, 1):
        query, keywords, dtype = q["query"], q["keywords"], q["doc_type"]

        t0 = time.time()
        chunks, scores = retrieve(model, index, records, query, prefix)
        answer = generate_answer(query, chunks)
        elapsed = time.time() - t0

        chunk_text = "\n".join(c["content"] for c in chunks)
        r_recall, _ = keyword_recall(chunk_text, keywords)
        a_recall, a_hits = keyword_recall(answer, keywords)
        judge = judge_answer(query, answer) if args.judge else None

        rows.append((r_recall, a_recall, elapsed, judge))
        by_type[dtype]["r"].append(r_recall)
        by_type[dtype]["a"].append(a_recall)
        if judge is not None:
            by_type[dtype]["j"].append(judge)

        top = f", top score {scores[0]:.2f}" if scores else ""
        print(f"\n[{i}] ({dtype}) {query}")
        print(f"    retrieved {len(chunks)} chunks{top}")
        print(f"    retrieval keyword recall : {r_recall:.0%}")
        print(f"    answer keyword recall    : {a_recall:.0%}  {a_hits}")
        print(f"    latency                  : {elapsed:.1f}s")
        if judge is not None:
            print(f"    LLM judge                : {judge}/5")
        print(f"    answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")

    n = len(rows)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    judges = [j for *_, j in rows if j is not None]

    print("\n" + "=" * 72)
    print("SUMMARY (mean over %d queries)" % n)
    print("=" * 72)
    print(f"Retrieval keyword recall : {mean([r for r, *_ in rows]):.0%}")
    print(f"Answer keyword recall    : {mean([a for _, a, *_ in rows]):.0%}")
    print(f"Avg latency              : {mean([t for _, _, t, _ in rows]):.1f}s")
    if judges:
        print(f"Avg LLM judge score      : {mean(judges):.1f}/5  (n={len(judges)})")

    print("\nBy document type:")
    print(f"  {'type':<20}{'retr.recall':>12}{'ans.recall':>12}{'judge':>8}")
    for dtype in sorted(by_type):
        d = by_type[dtype]
        j = f"{mean(d['j']):.1f}" if d["j"] else "-"
        print(f"  {dtype:<20}{mean(d['r']):>11.0%}{mean(d['a']):>12.0%}{j:>8}")


if __name__ == "__main__":
    main()
