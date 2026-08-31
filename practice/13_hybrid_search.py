# ============================================================
# 13_hybrid_search.py
# HYBRID SEARCH = VECTOR SEARCH + KEYWORD SEARCH
# ============================================================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi


# ============================================================
# 1. DOCUMENTS
# ============================================================

documents = [

    "The iPhone 15 has a 48MP main camera.",

    "The iPhone 15 uses the A16 Bionic chip.",

    "The iPhone 15 supports 5G connectivity.",

    "The iPhone 15 is available in 128GB, 256GB, and 512GB storage.",

    "The 128GB iPhone 15 costs ₹69,999.",

    "The 256GB iPhone 15 costs ₹79,999.",

    "The 512GB iPhone 15 costs ₹99,999.",

    "Customers can return the iPhone 15 within 7 days.",

    "Refunds are processed within 5-7 business days.",

    "The iPhone 15 supports satellite Emergency SOS."
]


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 3. CREATE VECTOR EMBEDDINGS
# ============================================================

print("\nCreating document embeddings...")

document_embeddings = embedding_model.encode(
    documents
)

print("Document embeddings created.")


# ============================================================
# 4. CREATE BM25 INDEX
# ============================================================

# BM25 works with words/tokens.
#
# Example:
#
# "iphone 15 camera"
#
# becomes:
#
# ["iphone", "15", "camera"]

tokenized_documents = [
    document.lower().split()
    for document in documents
]


bm25 = BM25Okapi(
    tokenized_documents
)


# ============================================================
# 5. HYBRID SEARCH FUNCTION
# ============================================================

def hybrid_search(query, k=5):

    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(query)

    vector_scores = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]


    # --------------------------------------------------------
    # KEYWORD SEARCH / BM25
    # --------------------------------------------------------

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )


    # --------------------------------------------------------
    # NORMALIZE BOTH SCORE TYPES
    # --------------------------------------------------------
    #
    # Vector similarity and BM25 scores have different scales.
    #
    # Therefore we normalize them before combining them.
    # --------------------------------------------------------

    def normalize(scores):

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:

            return [0.0 for _ in scores]

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]


    normalized_vector = normalize(
        vector_scores
    )

    normalized_bm25 = normalize(
        bm25_scores
    )


    # --------------------------------------------------------
    # COMBINE SCORES
    # --------------------------------------------------------
    #
    # 70% semantic/vector search
    # 30% keyword/BM25
    # --------------------------------------------------------

    vector_weight = 0.7

    keyword_weight = 0.3


    hybrid_scores = []

    for i in range(len(documents)):

        score = (
            vector_weight * normalized_vector[i]
            +
            keyword_weight * normalized_bm25[i]
        )

        hybrid_scores.append(score)


    # --------------------------------------------------------
    # SORT BY HYBRID SCORE
    # --------------------------------------------------------

    ranked_indices = sorted(
        range(len(documents)),
        key=lambda i: hybrid_scores[i],
        reverse=True
    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n===== HYBRID SEARCH RESULTS =====")

    for rank, index in enumerate(
        ranked_indices[:k],
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Vector Score  = "
            f"{vector_scores[index]:.3f}"
        )

        print(
            f"BM25 Score    = "
            f"{bm25_scores[index]:.3f}"
        )

        print(
            f"Hybrid Score  = "
            f"{hybrid_scores[index]:.3f}"
        )

        print(
            f"Document: {documents[index]}"
        )


# ============================================================
# 6. CHAT LOOP
# ============================================================

print("\n========================================")
print("         HYBRID SEARCH")
print("========================================")

print("\nAsk questions about the iPhone 15.")

print("Type 'exit' to quit.")


while True:

    query = input(
        "\nAsk a question: "
    ).strip()


    if query.lower() == "exit":

        print("\nGoodbye!")

        break


    if not query:

        print("Please enter a question.")

        continue


    hybrid_search(
        query,
        k=5
    )