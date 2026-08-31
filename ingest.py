"""PDF loading and chunking."""

import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# pypdf logs a warning per page when a PDF's /PageLabels table is malformed;
# we only use the numeric page index, so quiet it down.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def load_pdf(file_path):
    """Load a PDF into a list of page documents."""
    documents = PyPDFLoader(str(file_path)).load()
    if not any(doc.page_content.strip() for doc in documents):
        raise ValueError(
            "No extractable text found in the PDF (it may be scanned images)."
        )
    return documents


def split_text(documents):
    """Split page documents into overlapping character chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)
