from ingest import load_pdf, split_text
from vector_store import load_embedding_model, create_embeddings, store_in_faiss, save_index

# Load document
docs = load_pdf("data/Essay[1].pdf")

# Chunk
chunks = split_text(docs)

# Embeddings
model = load_embedding_model()
embeddings, texts = create_embeddings(model, chunks)

# FAISS
index = store_in_faiss(embeddings)

# SAVE (IMPORTANT)
save_index(index, texts)

print("Index built and saved successfully!")