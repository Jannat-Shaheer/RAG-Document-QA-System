import streamlit as st
import os

from ingest import load_pdf, split_text
from vector_store import (
    load_embedding_model,
    create_embeddings,
    store_in_faiss,
    save_index,
    load_index
)
from retriever import search
from generator import generate_answer   # reuse function

st.set_page_config(page_title="Document QA System", layout="wide")

st.title("Retrieval-Augmented AI for Document QA")

# Sidebar
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Cache Embedding Model
@st.cache_resource
def get_embedding_model():
    return load_embedding_model()

#Cache FAISS Index 
@st.cache_resource
def get_index():
    return load_index()

# PROCESS DOCUMENT (uses existing functions)
if uploaded_file is not None:
    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("File uploaded!")

    if st.sidebar.button("Process Document"):
        if "processed" not in st.session_state: #what happens if i try to process same doc again?
            st.session_state.processed = False

        with st.spinner("Processing..."):
            docs = load_pdf("data/temp.pdf")
            chunks = split_text(docs)

            model = load_embedding_model()
            embeddings, texts = create_embeddings(model, chunks)

            index = store_in_faiss(embeddings)
            save_index(index, texts)

        st.sidebar.success("Document processed!")
        st.session_state.processed = True

# LOAD INDEX (reuse)
if os.path.exists("vectorstore/index.faiss"):
    model = get_embedding_model()
    index, texts = get_index()

    st.subheader("Ask Questions")
    query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Thinking..."):
                # Use retriever function
                retrieved_docs = search(query, model, index, texts)

                # Use generator function
                answer = generate_answer(query, retrieved_docs)

            st.success("Answer:")
            st.write(answer)

            # Show sources
            st.subheader("Source Chunks")
            for i, doc in enumerate(retrieved_docs):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc)
                st.write("---")
        else:
            st.warning("Enter a question.")
else:
    st.info("Upload and process a document first.")