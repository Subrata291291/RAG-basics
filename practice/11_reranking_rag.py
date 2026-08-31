import os

from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# 2. CHECK API KEY
# ============================================================

if not api_key:

    print("ERROR: OPENROUTER_API_KEY not found.")
    print("Please check your .env file.")

    exit()


# ============================================================
# 3. CREATE OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 5. LOAD RERANKER MODEL
# ============================================================

print("\nLoading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker model loaded.")


# ============================================================
# 6. DOCUMENTS
# ============================================================

documents = [

    "The iPhone 15 has a 48MP main camera.",

    "The iPhone 15 uses the A16 Bionic chip.",

    "Customers can return the iPhone 15 within 7 days.",

    "The iPhone 15 supports 5G connectivity.",

    "The iPhone 15 is available in 128GB, 256GB, and 512GB storage.",

    "The 128GB iPhone 15 costs ₹69,999.",

    "The 256GB iPhone 15 costs ₹79,999.",

    "The 512GB iPhone 15 costs ₹99,999.",

    "The iPhone 15 is available in five colors.",

    "Refunds are processed within 5-7 business days.",

    "The iPhone 15 supports satellite connectivity through Emergency SOS in supported countries."

]


# ============================================================
# 7. CREATE DOCUMENT EMBEDDINGS
# ============================================================

print("\nCreating document embeddings...")

document_embeddings = embedding_model.encode(
    documents
)

print("Document embeddings created.")


# ============================================================
# 8. VECTOR SEARCH
# ============================================================

def vector_search(query, top_k=5):

    # --------------------------------------------------------
    # Convert query into embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(query)


    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]


    # --------------------------------------------------------
    # Sort documents by similarity
    # --------------------------------------------------------

    top_indices = similarities.argsort()[::-1][:top_k]


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results = []

    for index in top_indices:

        results.append({

            "document": documents[index],

            "vector_score": float(
                similarities[index]
            )

        })


    return results


# ============================================================
# 9. RERANK DOCUMENTS
# ============================================================

def rerank_documents(query, candidates):

    # --------------------------------------------------------
    # Create query-document pairs
    # --------------------------------------------------------

    pairs = [

        [query, item["document"]]

        for item in candidates

    ]


    # --------------------------------------------------------
    # Cross Encoder scores each pair
    # --------------------------------------------------------

    rerank_scores = reranker.predict(
        pairs
    )


    # --------------------------------------------------------
    # Add reranking score
    # --------------------------------------------------------

    for item, score in zip(
        candidates,
        rerank_scores
    ):

        item["rerank_score"] = float(score)


    # --------------------------------------------------------
    # Sort by reranker score
    # --------------------------------------------------------

    reranked = sorted(

        candidates,

        key=lambda x: x["rerank_score"],

        reverse=True

    )


    # --------------------------------------------------------
    # Return ALL reranked documents
    # --------------------------------------------------------

    return reranked


# ============================================================
# 10. GENERATE ANSWER
# ============================================================

def generate_answer(query, documents_for_answer):

    # --------------------------------------------------------
    # Create context
    # --------------------------------------------------------

    context = "\n\n".join(

        item["document"]

        for item in documents_for_answer

    )


    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:

"I don't have enough information in the provided documents."

Do not guess.
Do not use outside knowledge.

Context:
{context}

User question:
{query}

Answer clearly and briefly.
"""


    # --------------------------------------------------------
    # Send request to OpenRouter
    # --------------------------------------------------------

    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[

            {
                "role": "system",
                "content": (
                    "Answer questions using only "
                    "the provided documents."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )


    # --------------------------------------------------------
    # Extract answer
    # --------------------------------------------------------

    answer = response.choices[0].message.content

    return answer


# ============================================================
# 11. MAIN CHAT LOOP
# ============================================================

print("\n==============================================")
print("           RERANKING RAG CHATBOT")
print("==============================================")

print("\nAsk questions about the iPhone 15.")

print("Type 'exit' to quit.")


while True:

    # ========================================================
    # GET USER QUESTION
    # ========================================================

    query = input("\nAsk a question: ").strip()


    # ========================================================
    # EXIT
    # ========================================================

    if query.lower() == "exit":

        print("\nGoodbye!")

        break


    # ========================================================
    # EMPTY QUESTION
    # ========================================================

    if not query:

        print("Please enter a question.")

        continue


    # ========================================================
    # STEP 1 — VECTOR SEARCH
    # ========================================================

    print("\n===== VECTOR SEARCH =====")


    candidates = vector_search(

        query,

        top_k=5

    )


    # --------------------------------------------------------
    # Display vector search results
    # --------------------------------------------------------

    for rank, item in enumerate(

        candidates,

        start=1

    ):

        print(

            f"\nRank {rank}"
            f" | Vector Score = "
            f"{item['vector_score']:.3f}"

        )

        print(
            item["document"]
        )


    # ========================================================
    # STEP 2 — RERANKING
    # ========================================================

    print("\n===== RERANKING =====")


    reranked_documents = rerank_documents(

        query,

        candidates

    )


    # --------------------------------------------------------
    # Display ALL reranked documents
    # --------------------------------------------------------

    for rank, item in enumerate(

        reranked_documents,

        start=1

    ):

        print(

            f"\nRank {rank}"
            f" | Rerank Score = "
            f"{item['rerank_score']:.3f}"

        )

        print(
            item["document"]
        )


    # ========================================================
    # STEP 3 — SELECT TOP 3
    # ========================================================

    final_documents = reranked_documents[:3]


    print("\n===== DOCUMENTS SENT TO LLM =====")


    for rank, item in enumerate(

        final_documents,

        start=1

    ):

        print(

            f"\nDocument {rank}:"

        )

        print(
            item["document"]
        )


    # ========================================================
    # STEP 4 — GENERATE ANSWER
    # ========================================================

    print("\n===== GENERATING ANSWER =====")


    try:

        answer = generate_answer(

            query,

            final_documents

        )


        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------

        print("\n===== FINAL ANSWER =====")

        print(answer)


    except Exception as e:

        print("\nERROR while generating answer:")

        print(e)