from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Our documents
documents = [
    "The iPhone 15 has a 48MP main camera.",
    "The iPhone 15 uses the A16 Bionic chip.",
    "Customers can return the phone within 7 days.",
    "The iPhone 15 supports 5G connectivity.",
    "The iPhone 15 is available in 128GB, 256GB, and 512GB."
]


# User's question
query = "What camera does the iPhone 15 have?"


# Create embeddings
document_embeddings = model.encode(documents)
query_embedding = model.encode(query)


# Compare query with every document
similarities = cosine_similarity(
    [query_embedding],
    document_embeddings
)[0]


print("===== QUERY =====")
print(query)


print("\n===== RESULTS =====")

# Number of documents we want
k = 2


# Sort document indexes by similarity
top_indices = similarities.argsort()[::-1][:k]


print("\n===== TOP K DOCUMENTS =====")

for rank, index in enumerate(top_indices, start=1):

    print(
        f"Rank {rank} | "
        f"Document {index + 1} | "
        f"Similarity = {similarities[index]:.3f}"
    )

    print(documents[index])
    print()




"""
And now you have almost built the basic retrieval half of RAG:
                USER
                 │
                 ▼
        "What camera..."
                 │
                 ▼
          QUERY EMBEDDING
                 │
                 ▼
       ┌──────────────────┐
       │ Vector comparison│
       └──────────────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
       Doc1     Doc2     Doc3...
      0.769     0.665    0.206
        │
        ▼
       SORT
        │
        ▼
      TOP-K
        │
        ▼
  Retrieved Context
        │
        ▼
       LLM
"""