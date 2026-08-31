"""Answer generation with a local Ollama LLM."""

import ollama

import config

_NO_CONTEXT_MSG = "No relevant information found in the document."
_NOT_IN_DOC = "The document does not contain this information."

_PROMPT_TEMPLATE = """You answer questions using the provided context from a document.

Guidelines:
- Base the answer only on the context below. You may combine facts stated across
  several chunks; the answer does not need to appear verbatim.
- Only if the context has nothing relevant to the question, reply exactly:
  "{not_in_doc}"
- Be clear and concise (2-5 sentences); use a short list if the answer enumerates items.

Context:
{context}

Question:
{query}

Answer:"""


def _format_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page")
        tag = f"[chunk {i}" + (f", page {page + 1}]" if page is not None else "]")
        parts.append(f"{tag}\n{chunk['content']}")
    return "\n\n".join(parts)


def _build_prompt(query, chunks):
    return _PROMPT_TEMPLATE.format(
        not_in_doc=_NOT_IN_DOC,
        context=_format_context(chunks),
        query=query,
    )


def _error_message(exc):
    if isinstance(exc, ollama.ResponseError):
        return (
            f"LLM error: {exc.error}. Is the '{config.LLM_MODEL}' model pulled? "
            f"Try: ollama pull {config.LLM_MODEL}"
        )
    return "Cannot reach Ollama. Start it with `ollama serve` and try again."


def generate_answer(query, chunks):
    """Generate an answer from retrieved chunks (list of dicts with 'content')."""
    if not chunks:
        return _NO_CONTEXT_MSG
    try:
        response = ollama.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": _build_prompt(query, chunks)}],
        )
    except (ollama.ResponseError, ConnectionError) as e:
        return _error_message(e)
    return response["message"]["content"].strip()


def stream_answer(query, chunks):
    """Yield the answer incrementally, for a responsive UI (st.write_stream)."""
    if not chunks:
        yield _NO_CONTEXT_MSG
        return
    try:
        for part in ollama.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": _build_prompt(query, chunks)}],
            stream=True,
        ):
            yield part["message"]["content"]
    except (ollama.ResponseError, ConnectionError) as e:
        yield _error_message(e)
