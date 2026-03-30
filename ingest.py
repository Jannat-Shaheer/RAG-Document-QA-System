from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Step 2: Split text into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# if __name__ == "__main__":
#     file_path = "data/Essay[1].pdf"

#     docs = load_pdf(file_path)
#     chunks = split_text(docs)

#     print(f"Total chunks created: {len(chunks)}")
#     print("\nExample chunk:\n")
#     print(chunks[0].page_content)