from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 2. DOCUMENTS
# ==========================================

documents = [
    "The iPhone 15 has a 48MP main camera.",
    "The iPhone 15 uses the A16 Bionic chip.",
    "Customers can return the phone within 7 days.",
    "The iPhone 15 supports 5G connectivity.",
    "The iPhone 15 is available in 128GB, 256GB, and 512GB."
]


# ==========================================
# 3. CREATE DOCUMENT EMBEDDINGS
# ==========================================

# We only need to create these once.
document_embeddings = model.encode(documents)


# ==========================================
# 4. RETRIEVER FUNCTION
# ==========================================

def retrieve(query, k=2):

    # Convert the user's question into an embedding
    query_embedding = model.encode(query)

    # Compare query embedding with all document embeddings
    similarities = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]

    # Get indexes of the highest similarity scores
    top_indices = similarities.argsort()[::-1][:k]

    # Store the retrieved documents
    results = []

    for index in top_indices:

        results.append({
            "document": documents[index],
            "score": similarities[index]
        })

    return results


# ==========================================
# 5. INTERACTIVE QUESTION LOOP
# ==========================================

while True:

    query = input(
        "\nAsk a question "
        "(type 'exit' to quit): "
    )

    # Stop the program
    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    # Retrieve the most relevant documents
    results = retrieve(query, k=2)

    # Display results
    print("\n===== RETRIEVED DOCUMENTS =====")

    for rank, result in enumerate(results, start=1):

        print(
            f"\nRank {rank}"
            f" | Similarity = {result['score']:.3f}"
        )

        print(result["document"])