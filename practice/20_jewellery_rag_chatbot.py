# ============================================================
# 20_jewellery_rag_chatbot.py
#
# COMPLETE JEWELLERY RAG CHATBOT
#
# Pipeline:
#
# User Query
#     ↓
# Filter Extraction
#     ↓
# Metadata Filtering
#     ↓
# Semantic Search
#     ↓
# MMR
#     ↓
# OpenRouter LLM
#     ↓
# Final Answer
# ============================================================


import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


api_key = os.getenv(
    "OPENROUTER_API_KEY"
)


if not api_key:

    print(
        "\nERROR: OPENROUTER_API_KEY not found."
    )

    print(
        "Please check your .env file."
    )

    exit()


# ============================================================
# 2. OPENROUTER CLIENT
# ============================================================

client = OpenAI(

    base_url="https://openrouter.ai/api/v1",

    api_key=api_key
)


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print(
    "\nLoading embedding model..."
)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "Embedding model loaded."
)


# ============================================================
# 4. PRODUCT DATABASE
# ============================================================

products = [

    {
        "id": "R001",

        "name": "Classic 22K Gold Ring",

        "category": "ring",

        "metal": "gold",

        "karat": 22,

        "price": 18000,

        "description":
            "A classic 22K gold ring suitable "
            "for everyday wear."
    },


    {
        "id": "R002",

        "name": "Elegant 22K Gold Ring",

        "category": "ring",

        "metal": "gold",

        "karat": 22,

        "price": 25000,

        "description":
            "An elegant 22K gold ring with a "
            "beautiful traditional design, suitable "
            "for weddings and special occasions."
    },


    {
        "id": "R003",

        "name": "Traditional 22K Gold Ring",

        "category": "ring",

        "metal": "gold",

        "karat": 22,

        "price": 27000,

        "description":
            "A traditional 22K gold ring featuring "
            "an intricate Indian design, perfect "
            "for weddings and festive occasions."
    },


    {
        "id": "R004",

        "name": "Bridal 22K Gold Ring",

        "category": "ring",

        "metal": "gold",

        "karat": 22,

        "price": 29000,

        "description":
            "A beautiful bridal 22K gold ring with "
            "an elaborate traditional design made "
            "for wedding ceremonies."
    },


    {
        "id": "R005",

        "name": "Modern 22K Gold Ring",

        "category": "ring",

        "metal": "gold",

        "karat": 22,

        "price": 24000,

        "description":
            "A modern 22K gold ring with a minimal "
            "contemporary design suitable for "
            "parties and special occasions."
    },


    {
        "id": "R006",

        "name": "Diamond Gold Ring",

        "category": "ring",

        "metal": "gold",

        "karat": 18,

        "price": 19500,

        "description":
            "An 18K gold diamond ring with a modern "
            "design, ideal for special occasions "
            "and weddings."
    },


    {
        "id": "R007",

        "name": "Silver Ring",

        "category": "ring",

        "metal": "silver",

        "karat": 0,

        "price": 8000,

        "description":
            "A stylish silver ring with a simple "
            "modern design."
    },


    {
        "id": "N001",

        "name": "22K Gold Necklace",

        "category": "necklace",

        "metal": "gold",

        "karat": 22,

        "price": 35000,

        "description":
            "A beautiful 22K gold necklace suitable "
            "for festive occasions and weddings."
    },


    {
        "id": "N002",

        "name": "Diamond Gold Necklace",

        "category": "necklace",

        "metal": "gold",

        "karat": 18,

        "price": 45000,

        "description":
            "An 18K gold necklace featuring diamond "
            "detailing for weddings and special occasions."
    },


    {
        "id": "B001",

        "name": "22K Gold Bracelet",

        "category": "bracelet",

        "metal": "gold",

        "karat": 22,

        "price": 22000,

        "description":
            "A 22K gold bracelet with a traditional "
            "design suitable for festive occasions."
    },


    {
        "id": "E001",

        "name": "Gold Earrings",

        "category": "earrings",

        "metal": "gold",

        "karat": 22,

        "price": 15000,

        "description":
            "22K gold earrings with an elegant design "
            "suitable for weddings and festive occasions."
    }
]


# ============================================================
# 5. CONVERT PRODUCT TO TEXT
# ============================================================

def product_to_text(product):

    return f"""
Product: {product['name']}

Category: {product['category']}

Metal: {product['metal']}

Karat: {product['karat']}K

Price: ₹{product['price']}

Description:
{product['description']}
"""


# ============================================================
# 6. CREATE PRODUCT EMBEDDINGS
# ============================================================

print(
    "\nCreating product embeddings..."
)


product_texts = [

    product_to_text(product)

    for product in products
]


product_embeddings = embedding_model.encode(
    product_texts
)


print(
    "Product embeddings created."
)


# ============================================================
# 7. PRICE PARSER
# ============================================================

def parse_price(
    value,
    unit=None
):

    value = value.replace(
        ",",
        ""
    )


    number = float(value)


    if unit:

        if unit.lower() == "k":

            number *= 1000


    return int(number)


# ============================================================
# 8. EXTRACT FILTERS
# ============================================================

def extract_filters(query):

    query = query.lower().strip()


    filters = {

        "category": None,

        "metal": None,

        "karat": None,

        "max_price": None,

        "min_price": None
    }


    # ========================================================
    # CATEGORY
    # ========================================================

    if re.search(
        r"\brings?\b",
        query
    ):

        filters["category"] = "ring"


    elif re.search(
        r"\bnecklaces?\b",
        query
    ):

        filters["category"] = "necklace"


    elif re.search(
        r"\bbracelets?\b",
        query
    ):

        filters["category"] = "bracelet"


    elif re.search(
        r"\b(?:earrings?|ear\s*rings?)\b",
        query
    ):

        filters["category"] = "earrings"


    # ========================================================
    # METAL
    # ========================================================

    if re.search(
        r"\bgold\b",
        query
    ):

        filters["metal"] = "gold"


    elif re.search(
        r"\bsilver\b",
        query
    ):

        filters["metal"] = "silver"


    # ========================================================
    # KARAT
    # ========================================================

    karat_match = re.search(

        r"\b(18|20|22|24)\s*k(?:arat)?\b",

        query
    )


    if karat_match:

        filters["karat"] = int(
            karat_match.group(1)
        )


    # ========================================================
    # MAX PRICE
    # ========================================================

    max_patterns = [

        r"\bunder\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bbelow\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bless\s+than\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bupto\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bup\s+to\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?"
    ]


    for pattern in max_patterns:

        match = re.search(
            pattern,
            query
        )


        if match:

            filters["max_price"] = parse_price(

                match.group(1),

                match.group(2)
            )

            break


    # ========================================================
    # MIN PRICE
    # ========================================================

    min_patterns = [

        r"\babove\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bover\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bmore\s+than\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?"
    ]


    for pattern in min_patterns:

        match = re.search(
            pattern,
            query
        )


        if match:

            filters["min_price"] = parse_price(

                match.group(1),

                match.group(2)
            )

            break


    return filters


# ============================================================
# 9. METADATA FILTER
# ============================================================

def metadata_filter(filters):

    results = []


    for product in products:


        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        if filters["category"] is not None:

            if product["category"] != filters["category"]:

                continue


        # ----------------------------------------------------
        # METAL
        # ----------------------------------------------------

        if filters["metal"] is not None:

            if product["metal"] != filters["metal"]:

                continue


        # ----------------------------------------------------
        # KARAT
        # ----------------------------------------------------

        if filters["karat"] is not None:

            if product["karat"] != filters["karat"]:

                continue


        # ----------------------------------------------------
        # MAX PRICE
        # ----------------------------------------------------

        if filters["max_price"] is not None:

            if product["price"] > filters["max_price"]:

                continue


        # ----------------------------------------------------
        # MIN PRICE
        # ----------------------------------------------------

        if filters["min_price"] is not None:

            if product["price"] < filters["min_price"]:

                continue


        results.append(
            product
        )


    return results


# ============================================================
# 10. MMR SEARCH
# ============================================================

def mmr_search(

    query,

    filtered_products,

    top_k=3,

    lambda_value=0.7
):

    if not filtered_products:

        return []


    # --------------------------------------------------------
    # Find original embedding indexes
    # --------------------------------------------------------

    filtered_indices = [

        products.index(product)

        for product in filtered_products
    ]


    candidate_embeddings = product_embeddings[
        filtered_indices
    ]


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query
    )


    # --------------------------------------------------------
    # Query similarity
    # --------------------------------------------------------

    query_scores = cosine_similarity(

        [query_embedding],

        candidate_embeddings

    )[0]


    # --------------------------------------------------------
    # Document similarity
    # --------------------------------------------------------

    document_similarities = cosine_similarity(

        candidate_embeddings

    )


    # --------------------------------------------------------
    # MMR
    # --------------------------------------------------------

    selected = []

    remaining = list(
        range(
            len(filtered_products)
        )
    )


    while (

        remaining

        and

        len(selected) < top_k

    ):

        best_candidate = None

        best_score = float("-inf")


        for candidate in remaining:


            # ------------------------------------------------
            # Relevance
            # ------------------------------------------------

            relevance = query_scores[
                candidate
            ]


            # ------------------------------------------------
            # Redundancy
            # ------------------------------------------------

            if not selected:

                redundancy = 0


            else:

                redundancy = max(

                    document_similarities[
                        candidate
                    ][selected]

                )


            # ------------------------------------------------
            # MMR SCORE
            # ------------------------------------------------

            mmr_score = (

                lambda_value * relevance

                -

                (1 - lambda_value)
                * redundancy

            )


            # ------------------------------------------------
            # Best candidate
            # ------------------------------------------------

            if mmr_score > best_score:

                best_score = mmr_score

                best_candidate = candidate


        # ----------------------------------------------------
        # Select
        # ----------------------------------------------------

        selected.append(
            best_candidate
        )


        remaining.remove(
            best_candidate
        )


    # ========================================================
    # BUILD RESULT
    # ========================================================

    results = []


    for index in selected:

        results.append({

            "product":
                filtered_products[index],

            "query_score":
                float(query_scores[index]),

            "mmr_score":
                None
        })


    return results


# ============================================================
# 11. GENERATE LLM ANSWER
# ============================================================

def generate_answer(

    query,

    results
):

    # --------------------------------------------------------
    # Create context
    # --------------------------------------------------------

    context_parts = []


    for item in results:

        product = item["product"]


        context_parts.append(

            f"""
Product ID: {product['id']}

Name: {product['name']}

Category: {product['category']}

Metal: {product['metal']}

Karat: {product['karat']}K

Price: ₹{product['price']:,}

Description:
{product['description']}
"""
        )


    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful jewellery shopping assistant.

Answer the user's question using ONLY the
product information provided in the context.

IMPORTANT RULES:

1. Do not invent products.
2. Do not invent prices.
3. Do not invent karat values.
4. Do not invent product features.
5. Do not claim that a product has a feature
   unless the context explicitly says so.
6. If the requested information is not present,
   clearly say that the available product data
   does not provide that information.
7. Keep the answer concise and useful.
8. If multiple products are relevant, mention
   the best matching products.
9. When discussing prices, use ₹.
10. Do not mention embeddings, vectors, MMR,
    metadata filtering, or internal processing.

USER QUESTION:

{query}


PRODUCT CONTEXT:

{context}


ANSWER:
"""


    # --------------------------------------------------------
    # OpenRouter
    # --------------------------------------------------------

    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[

            {
                "role": "system",

                "content":
                    "You are a helpful jewellery "
                    "shopping assistant. "
                    "Use only the supplied product context."
            },

            {
                "role": "user",

                "content": prompt
            }

        ]
    )


    # --------------------------------------------------------
    # Return answer
    # --------------------------------------------------------

    return response.choices[
        0
    ].message.content


# ============================================================
# 12. DISPLAY FILTERS
# ============================================================

def display_filters(filters):

    print(
        "\n===== DETECTED FILTERS ====="
    )


    for key, value in filters.items():

        print(
            f"{key:12} → {value}"
        )


# ============================================================
# 13. DISPLAY MMR RESULTS
# ============================================================

def display_results(results):

    print(
        "\n===== FINAL RETRIEVED PRODUCTS ====="
    )


    for rank, item in enumerate(
        results,
        start=1
    ):

        product = item["product"]


        print(
            f"\nRank {rank}"
        )


        print(
            f"Semantic Score = "
            f"{item['query_score']:.3f}"
        )


        print(
            f"ID: {product['id']}"
        )


        print(
            f"Name: {product['name']}"
        )


        print(
            f"Price: ₹{product['price']:,}"
        )


        print(
            f"Description: "
            f"{product['description']}"
        )


# ============================================================
# 14. COMPLETE SEARCH PIPELINE
# ============================================================

def search(query):

    # ========================================================
    # STEP 1 — EXTRACT FILTERS
    # ========================================================

    filters = extract_filters(
        query
    )


    display_filters(
        filters
    )


    # ========================================================
    # STEP 2 — METADATA FILTER
    # ========================================================

    filtered_products = metadata_filter(
        filters
    )


    print(
        "\n===== METADATA FILTER ====="
    )


    print(
        f"Matching products: "
        f"{len(filtered_products)}"
    )


    for product in filtered_products:

        print(

            f"- {product['name']} "
            f"| ₹{product['price']:,}"
        )


    if not filtered_products:

        print(
            "\nNo products match your "
            "specified requirements."
        )

        return


    # ========================================================
    # STEP 3 — MMR
    # ========================================================

    results = mmr_search(

        query,

        filtered_products,

        top_k=3,

        lambda_value=0.7
    )


    # ========================================================
    # STEP 4 — DISPLAY RESULTS
    # ========================================================

    display_results(
        results
    )


    # ========================================================
    # STEP 5 — LLM
    # ========================================================

    print(
        "\n===== GENERATING ANSWER ====="
    )


    try:

        answer = generate_answer(

            query,

            results
        )


        print(
            "\n===== FINAL ANSWER ====="
        )


        print(
            answer
        )


    except Exception as e:

        print(
            "\nERROR FROM OPENROUTER:"
        )


        print(
            e
        )


# ============================================================
# 15. CHAT LOOP
# ============================================================

print(
    "\n=============================================="
)

print(
    "          JEWELLERY RAG CHATBOT"
)

print(
    "=============================================="
)


print(
    "\nYou can ask questions about the jewellery."
)


print(
    "\nExamples:"
)


print(
    "  Show me a beautiful 22K gold ring under 30k"
)


print(
    "  I want a traditional gold ring for wedding"
)


print(
    "  Show me a modern gold ring"
)


print(
    "  I want something under 20k"
)


print(
    "\nType 'exit' to quit."
)


while True:

    query = input(
        "\nAsk your jewellery question: "
    ).strip()


    # ========================================================
    # EXIT
    # ========================================================

    if query.lower() == "exit":

        print(
            "\nGoodbye!"
        )

        break


    # ========================================================
    # EMPTY QUERY
    # ========================================================

    if not query:

        print(
            "Please enter a question."
        )

        continue


    # ========================================================
    # SEARCH
    # ========================================================

    search(
        query
    )



"""
                 USER
                   │
                   ▼
           Natural Language
                   │
                   ▼
        ┌────────────────────┐
        │ Query Understanding│
        └─────────┬──────────┘
                  │
          category / metal
          karat / price
                  │
                  ▼
        ┌────────────────────┐
        │ Metadata Filtering │
        └─────────┬──────────┘
                  │
             Valid Products
                  │
                  ▼
        ┌────────────────────┐
        │ Semantic Search    │
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │       MMR          │
        │ Relevance+Variety  │
        └─────────┬──────────┘
                  │
                  ▼
           Top Products
                  │
                  ▼
        ┌────────────────────┐
        │    OpenRouter LLM  │
        └─────────┬──────────┘
                  │
                  ▼
           Natural Answer
"""