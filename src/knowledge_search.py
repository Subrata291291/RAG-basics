import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings import (
    embed_text,
    embed_texts
)


# ============================================================
# BUILD KNOWLEDGE EMBEDDINGS
# ============================================================

def build_knowledge_embeddings(
    documents
):

    texts = [
        document["content"]
        for document in documents
    ]

    return embed_texts(texts)


# ============================================================
# SEARCH KNOWLEDGE
# ============================================================

def search_knowledge(
    query,
    documents,
    embeddings,
    top_k=3
):

    if not documents:

        return []

    query_embedding = embed_text(query)

    scores = cosine_similarity(
        [query_embedding],
        embeddings
    )[0]

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "document": documents[index],
                "score": float(scores[index])
            }
        )

    return results