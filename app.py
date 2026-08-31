import streamlit as st

import config
from ingest import load_pdf, split_text
from vector_store import (
    load_embedding_model,
    create_embeddings,
    store_in_faiss,
    extend_index,
    save_index,
    load_index,
    index_exists,
)
from retriever import search
from generator import stream_answer

st.set_page_config(page_title="Document QA System", layout="wide")
st.title("Retrieval-Augmented AI for Document QA")

config.ensure_dirs()
st.session_state.setdefault("processed", False)
st.session_state.setdefault("last_file", None)


@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource
def get_index():
    return load_index()


# --- Sidebar: upload + process --------------------------------------------
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    config.UPLOAD_PATH.write_bytes(uploaded_file.read())
    st.sidebar.success("File uploaded!")

    if st.session_state.last_file != uploaded_file.name:
        st.session_state.processed = False
        st.session_state.last_file = uploaded_file.name

    mode = st.sidebar.radio(
        "When processing",
        ["Replace index", "Add to index"],
        help="Add to index keeps previously processed documents searchable.",
        disabled=not index_exists(),
    )

    if st.sidebar.button("Process Document", disabled=st.session_state.processed):
        try:
            with st.spinner("Processing..."):
                chunks = split_text(load_pdf(config.UPLOAD_PATH))
                model = get_embedding_model()
                embeddings, records = create_embeddings(model, chunks)
                if mode == "Add to index" and index_exists():
                    _, all_records = extend_index(embeddings, records)
                else:
                    save_index(store_in_faiss(embeddings), records)
                    all_records = records
            get_index.clear()  # drop the stale cached index
            st.session_state.processed = True
            st.sidebar.success(
                f"Indexed {len(records)} new chunks ({len(all_records)} total)"
            )
            st.rerun()
        except (ValueError, RuntimeError) as e:
            st.sidebar.error(f"Could not process document: {e}")
    elif st.session_state.processed:
        st.sidebar.success("Document processed")

# --- Main: ask questions -------------------------------------------------------
if not index_exists():
    st.info("Upload and process a document first.")
    st.stop()

model = get_embedding_model()
index, records = get_index()

st.subheader("Ask Questions")
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Enter a question.")
        st.stop()

    with st.spinner("Retrieving..."):
        chunks, scores = search(query, model, index, records)

    st.success("Answer:")
    with st.spinner("Generating — the model reads the context first; "
                    "first tokens can take 1-2 min on CPU."):
        st.write_stream(stream_answer(query, chunks))

    if chunks:
        st.subheader("Source Chunks")
        for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
            page = chunk.get("page")
            source = chunk.get("source")
            label = f"Chunk {i} — similarity {score:.2f}"
            if source:
                label += f" — {source}"
            if page is not None:
                label += f" p.{page + 1}"
            with st.expander(label):
                st.write(chunk["content"])
