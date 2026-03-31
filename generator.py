import ollama

# Step 1: Generate answer using LLM
def generate_answer(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    #push to git the model change to phi
    response = ollama.chat(
        #model='mistral',
        model='phi',
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']


# if __name__ == "__main__":

#     # Load model
#     model = load_embedding_model()

#     # Load FAISS (instead of rebuilding)
#     index, texts = load_index()

#     # Query
#     query = "What is the applicant's applying for?"

#     # Retrieve
#     retrieved_docs = search(query, model, index, texts)

#     # Generate answer
#     answer = generate_answer(query, retrieved_docs)

#     print("\nQuestion:", query)
#     print("\nAnswer:\n", answer)