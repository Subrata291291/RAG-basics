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

for i, score in enumerate(similarities):

    print(
        f"Document {i + 1} | "
        f"Similarity = {score:.3f}"
    )

    print(documents[i])
    print()




"""
Right now our program does this:

Query
 ↓
Create query embedding
 ↓
Compare with all documents
 ↓
Print similarity scores


But a real RAG system needs:

Query
 ↓
Embeddings
 ↓
Similarity
 ↓
SORT
 ↓
TOP-K
 ↓
Retrieved chunks
 ↓
LLM
"""