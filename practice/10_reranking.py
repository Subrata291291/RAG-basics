import os

from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer
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

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 5. DOCUMENTS
# ============================================================

documents = [

    "The iPhone 15 has a 48MP main camera.",

    "The iPhone 15 uses the A16 Bionic chip.",

    "The iPhone 15 supports 5G connectivity.",

    "The iPhone 15 is available in 128GB, 256GB and 512GB storage.",

    "Customers can return the phone within 7 days.",

    "Satellite Emergency SOS is available on iPhone 14 and later in supported countries."

]


# ============================================================
# 6. CREATE DOCUMENT EMBEDDINGS
# ============================================================

print("Creating document embeddings...")

document_embeddings = embedding_model.encode(
    documents
)

print("Document embeddings created.")


# ============================================================
# 7. VECTOR RETRIEVAL
# ============================================================

def retrieve_documents(query, k=5):

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
    # Sort by similarity
    # --------------------------------------------------------

    top_indices = similarities.argsort()[::-1][:k]


    # --------------------------------------------------------
    # Create results
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
# 8. LLM RERANKER
# ============================================================

def rerank_documents(query, results):

    # --------------------------------------------------------
    # Prepare documents for the LLM
    # --------------------------------------------------------

    documents_text = "\n\n".join(

        [
            f"Document {i + 1}:\n{item['document']}"

            for i, item in enumerate(results)
        ]

    )


    # --------------------------------------------------------
    # Reranking prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document relevance evaluator.

Your job is to determine how relevant each document
is for answering the user's question.

User question:

{query}


Retrieved documents:

{documents_text}


Give every document a relevance score from 0 to 10.

Scoring rules:

10 = Directly answers the question
7-9 = Highly relevant
4-6 = Somewhat related
1-3 = Weakly related
0 = Completely irrelevant


Return ONLY this format:

Document 1: score
Document 2: score
Document 3: score
Document 4: score
Document 5: score

Do not provide explanations.
"""


    # --------------------------------------------------------
    # Send request to OpenRouter
    # --------------------------------------------------------

    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[

            {
                "role": "system",
                "content": "You are a document relevance evaluator."
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )


    # --------------------------------------------------------
    # Get LLM response
    # --------------------------------------------------------

    result = response.choices[0].message.content

    return result


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

print()
print("==========================================")
print("          RAG RERANKING PRACTICE")
print("==========================================")

print()
print("Ask a question about the iPhone 15.")
print("Type 'exit' to quit.")


while True:

    # --------------------------------------------------------
    # Get user question
    # --------------------------------------------------------

    query = input("\nAsk a question: ").strip()


    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if query.lower() == "exit":

        print("\nGoodbye!")

        break


    # --------------------------------------------------------
    # Ignore empty question
    # --------------------------------------------------------

    if not query:

        print("Please enter a question.")

        continue


    # ========================================================
    # STEP 1 — VECTOR SEARCH
    # ========================================================

    results = retrieve_documents(
        query,
        k=5
    )


    # ========================================================
    # DISPLAY VECTOR RESULTS
    # ========================================================

    print()
    print("===== VECTOR SEARCH RESULTS =====")


    for rank, item in enumerate(
        results,
        start=1
    ):

        print()

        print(
            f"Rank {rank}"
            f" | Vector Score = "
            f"{item['vector_score']:.3f}"
        )

        print(
            item["document"]
        )


    # ========================================================
    # STEP 2 — RERANK
    # ========================================================

    print()
    print("===== RERANKING =====")


    try:

        reranked_result = rerank_documents(
            query,
            results
        )


        print()
        print(reranked_result)


    except Exception as e:

        print()
        print("ERROR DURING RERANKING:")

        print(e)