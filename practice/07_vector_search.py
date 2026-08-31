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

document_embeddings = model.encode(documents)


# ==========================================
# 4. NUMBER OF RESULTS
# ==========================================

k = 2


# ==========================================
# 5. ASK QUESTIONS MULTIPLE TIMES
# ==========================================

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    # Stop the program
    if query.lower() == "exit":
        print("\nGoodbye!")
        break


    # ==========================================
    # 6. CREATE QUERY EMBEDDING
    # ==========================================

    query_embedding = model.encode(query)


    # ==========================================
    # 7. CALCULATE SIMILARITY
    # ==========================================

    similarities = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]


    # ==========================================
    # 8. FIND TOP-K DOCUMENTS
    # ==========================================

    top_indices = similarities.argsort()[::-1][:k]


    # ==========================================
    # 9. DISPLAY RESULTS
    # ==========================================

    print("\n===== TOP RESULTS =====")

    for rank, index in enumerate(top_indices, start=1):

        print(
            f"\nRank {rank}"
            f" | Similarity = {similarities[index]:.3f}"
        )

        print(documents[index])


"""
                USER
                  │
                  ▼
            User Question
                  │
                  ▼
          Query Embedding
                  │
                  ▼
       ┌─────────────────────┐
       │ Cosine Similarity   │
       │                     │
       │ Query ↔ Documents   │
       └─────────────────────┘
                  │
                  ▼
             Sort Scores
                  │
                  ▼
                TOP-K
                  │
                  ▼
        Relevant Documents
                  │
                  │
                  └───────┐
                          │
                    Ask Again
                          │
                          ▼
                         LOOP
"""

