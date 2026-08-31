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

    "Customers can return the iPhone 15 within 7 days.",

    "The iPhone 15 supports 5G connectivity.",

    "The iPhone 15 is available in 128GB, 256GB, and 512GB storage.",

    "The 128GB iPhone 15 costs ₹69,999.",

    "The 256GB iPhone 15 costs ₹79,999.",

    "The 512GB iPhone 15 costs ₹99,999.",

    "The iPhone 15 is available in five colors.",

    "Refunds are processed within 5-7 business days."
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
# 7. RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(query, k=2, threshold=0.70):

    # --------------------------------------------------------
    # Convert user query into embedding
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

    top_indices = similarities.argsort()[::-1][:k]

    # --------------------------------------------------------
    # Keep only relevant documents
    # --------------------------------------------------------

    results = []

    for index in top_indices:

        score = similarities[index]

        if score >= threshold:

            results.append({
                "document": documents[index],
                "score": score
            })

    return results

# ============================================================
# 8. GENERATOR FUNCTION
# ============================================================

def generate_answer(query, retrieved_documents):

    # --------------------------------------------------------
    # Create RAG context
    # --------------------------------------------------------

    context = "\n\n".join(
        [
            f"Document {i + 1}: {item['document']}"
            for i, item in enumerate(retrieved_documents)
        ]
    )


    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:

"I don't have enough information in the provided documents."

Do not invent information.

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
                "content": "You answer questions using retrieved documents."
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
# 9. MAIN CHAT LOOP
# ============================================================

print("\n========================================")
print("        BASIC RAG CHATBOT")
print("========================================")

print("\nAsk questions about the iPhone 15.")

print("Type 'exit' to quit.")


while True:

    # --------------------------------------------------------
    # Get user question
    # --------------------------------------------------------

    query = input("\nAsk a question: ").strip()


    # --------------------------------------------------------
    # Exit condition
    # --------------------------------------------------------

    if query.lower() == "exit":

        print("\nGoodbye!")

        break


    # --------------------------------------------------------
    # Ignore empty questions
    # --------------------------------------------------------

    if not query:

        print("Please enter a question.")

        continue


    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieved_documents = retrieve_documents(
        query,
        k=2,
        threshold=0.70
    )

    if not retrieved_documents:

        print("\n===== FINAL AI GENERATED ANSWER =====")

        print(
            "I don't have enough information "
            "in the provided documents."
        )

        continue


    # ========================================================
    # DISPLAY RETRIEVED DOCUMENTS
    # ========================================================

    print("\n===== RETRIEVED DOCUMENTS =====")

    for rank, item in enumerate(
        retrieved_documents,
        start=1
    ):

        print(
            f"\nRank {rank}"
            f" | Similarity = {item['score']:.3f}"
        )

        print(item["document"])


    # ========================================================
    # GENERATION
    # ========================================================

    print("\n===== GENERATING ANSWER =====")

    try:

        answer = generate_answer(
            query,
            retrieved_documents
        )

        print("\n===== FINAL ANSWER =====")

        print(answer)

    except Exception as e:

        print("\nERROR while generating answer:")

        print(e)