from ingest import load_pdf, split_text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load embedding model
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Create embeddings
def create_embeddings(model, chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings, texts

# Store in FAISS
def store_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

#  SAVE INDEX
def save_index(index, texts):
    faiss.write_index(index, "vectorstore/index.faiss")
    with open("vectorstore/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

#  LOAD INDEX
def load_index():
    index = faiss.read_index("vectorstore/index.faiss")
    with open("vectorstore/texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts

# if __name__ == "__main__":
#     # Load and split document
#     file_path = "data/Essay[1].pdf"

#     docs = load_pdf(file_path)
#     chunks = split_text(docs)

#     # Load embedding model
#     model = load_embedding_model()

#     # Create embeddings
#     embeddings, texts = create_embeddings(model, chunks)

#     # Store in FAISS
#     index = store_in_faiss(embeddings)

#     print("Embeddings shape:", embeddings.shape)
#     print("FAISS index size:", index.ntotal)