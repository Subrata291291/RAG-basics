from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 2. DOCUMENTS
# ============================================================

documents = [

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    "The iPhone 15 has a 48MP main camera.",

    "The iPhone 15 features a powerful 48MP camera.",

    "The iPhone 15 camera has 48 megapixels.",


    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    "The iPhone 15 uses the A16 Bionic chip.",

    "The A16 Bionic chip powers the iPhone 15.",


    # --------------------------------------------------------
    # CONNECTIVITY
    # --------------------------------------------------------

    "The iPhone 15 supports 5G connectivity.",

    "The iPhone 15 provides fast 5G network support.",


    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    "The iPhone 15 is available in 128GB, 256GB, and 512GB.",

    "The iPhone 15 comes with multiple storage options."
]


# ============================================================
# 3. USER QUERY
# ============================================================

query = input(
    "\nAsk a question: "
)


# ============================================================
# 4. CREATE DOCUMENT EMBEDDINGS
# ============================================================

print("\nCreating document embeddings...")

document_embeddings = model.encode(
    documents
)

print("Document embeddings created.")


# ============================================================
# 5. CREATE QUERY EMBEDDING
# ============================================================

query_embedding = model.encode(
    query
)


# ============================================================
# 6. QUERY → DOCUMENT SIMILARITY
# ============================================================

query_similarities = cosine_similarity(
    [query_embedding],
    document_embeddings
)[0]


# ============================================================
# 7. NORMAL VECTOR SEARCH
# ============================================================

k = 5

top_indices = query_similarities.argsort()[::-1][:k]


print("\n==========================================")
print("        NORMAL VECTOR SEARCH")
print("==========================================")


for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"\nRank {rank}"
        f" | Similarity = "
        f"{query_similarities[index]:.3f}"
    )

    print(documents[index])


# ============================================================
# 8. MMR FUNCTION
# ============================================================

def mmr_search(
    query_embedding,
    document_embeddings,
    documents,
    candidate_indices,
    k=3,
    lambda_param=0.3
):

    selected_indices = []


    # --------------------------------------------------------
    # We only work with candidates returned by vector search
    # --------------------------------------------------------

    remaining_indices = list(
        candidate_indices
    )


    # ========================================================
    # FIRST DOCUMENT
    # ========================================================

    first_index = max(
        remaining_indices,
        key=lambda index:
        query_similarities[index]
    )


    selected_indices.append(
        first_index
    )

    remaining_indices.remove(
        first_index
    )


    print("\n==========================================")
    print("              MMR SELECTION")
    print("==========================================")


    print("\nFirst document selected:")

    print(
        f"Query Similarity = "
        f"{query_similarities[first_index]:.3f}"
    )

    print(
        documents[first_index]
    )


    # ========================================================
    # SELECT REMAINING DOCUMENTS
    # ========================================================

    while (
        len(selected_indices) < k
        and remaining_indices
    ):

        best_mmr_score = float("-inf")

        best_index = None

        best_relevance = None

        best_redundancy = None


        print("\n------------------------------------------")
        print("Evaluating remaining candidates...")
        print("------------------------------------------")


        for candidate_index in remaining_indices:

            # =================================================
            # RELEVANCE
            # =================================================

            relevance = query_similarities[
                candidate_index
            ]


            # =================================================
            # REDUNDANCY
            # =================================================

            selected_embeddings = (
                document_embeddings[
                    selected_indices
                ]
            )


            candidate_embedding = (
                document_embeddings[
                    candidate_index
                ]
            )


            similarities_to_selected = (
                cosine_similarity(
                    [candidate_embedding],
                    selected_embeddings
                )[0]
            )


            redundancy = max(
                similarities_to_selected
            )


            # =================================================
            # MMR SCORE
            # =================================================

            mmr_score = (
                lambda_param * relevance
                -
                (1 - lambda_param) * redundancy
            )


            print(
                f"\nCandidate: "
                f"{documents[candidate_index]}"
            )

            print(
                f"Relevance   = {relevance:.3f}"
            )

            print(
                f"Redundancy  = {redundancy:.3f}"
            )

            print(
                f"MMR Score   = {mmr_score:.3f}"
            )


            # =================================================
            # KEEP BEST CANDIDATE
            # =================================================

            if mmr_score > best_mmr_score:

                best_mmr_score = mmr_score

                best_index = candidate_index

                best_relevance = relevance

                best_redundancy = redundancy


        # ====================================================
        # ADD WINNER
        # ====================================================

        selected_indices.append(
            best_index
        )

        remaining_indices.remove(
            best_index
        )


        print("\n>>> SELECTED")

        print(
            f"MMR Score = "
            f"{best_mmr_score:.3f}"
        )

        print(
            documents[best_index]
        )


    return selected_indices


# ============================================================
# 9. RUN MMR
# ============================================================

mmr_indices = mmr_search(

    query_embedding,

    document_embeddings,

    documents,

    candidate_indices=top_indices,

    k=3,

    # Higher value means:
    # MORE importance to relevance
    # LESS importance to diversity

    lambda_param=0.3
)


# ============================================================
# 10. FINAL MMR RESULTS
# ============================================================

print("\n==========================================")
print("              FINAL MMR RESULTS")
print("==========================================")


for rank, index in enumerate(
    mmr_indices,
    start=1
):

    print(
        f"\nRank {rank}"
    )

    print(
        f"Query Similarity = "
        f"{query_similarities[index]:.3f}"
    )

    print(
        documents[index]
    )


# ============================================================
# 11. CONCEPT
# ============================================================

"""
==============================================================
                    MMR
==============================================================


VECTOR SEARCH
=============

User Query
     |
     v
Embedding
     |
     v
Compare against documents
     |
     v
Top-K candidates


Example:

Camera Query

Camera document       0.636
Camera document       0.630
Camera document       0.620
Chip document         0.680
5G document           0.591


==============================================================

MMR
===

MMR asks:

1. Is this document relevant to the query?

AND

2. Is this document too similar to something
   we already selected?


==============================================================

FORMULA
=======

MMR =
λ × Relevance
-
(1 - λ) × Redundancy


==============================================================

lambda = 0.7

Means:

70% importance → Relevance

30% importance → Diversity


==============================================================

IMPORTANT
=========

MMR does NOT replace a good retrieval model.

If vector search says:

Chip = 0.680
Camera = 0.636

MMR starts from that ranking.

Therefore:

MMR is mainly used to reduce
REDUNDANCY.

Reranking is used to improve
RELEVANCE.


==============================================================
"""