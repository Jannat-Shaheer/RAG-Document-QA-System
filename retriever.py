from ingest import load_pdf, split_text
from vector_store import load_embedding_model, create_embeddings, store_in_faiss
import numpy as np

# Step 1: Search function
def search(query, model, index, texts, k=3):
    # Convert query → embedding
    query_vector = model.encode([query])

    # Search in FAISS
    distances, indices = index.search(np.array(query_vector), k)

    # Retrieve top-k results
    results = [texts[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    # Load and process document
    file_path = "data/Essay[1].pdf"

    docs = load_pdf(file_path)
    chunks = split_text(docs)

    # Embeddings
    model = load_embedding_model()
    embeddings, texts = create_embeddings(model, chunks)

    # FAISS
    index = store_in_faiss(embeddings)

    # Test query
    query = "What is the applicant's academic background?"
    results = search(query, model, index, texts)

    print("\nQuery:", query)
    print("\nTop results:\n")

    for i, res in enumerate(results):
        print(f"{i+1}. {res}\n")