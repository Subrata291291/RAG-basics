# ============================================================
# 17_hybrid_jewellery_rag.py
#
# HYBRID JEWELLERY RAG
#
# Metadata Filtering
# +
# Semantic Search
# +
# OpenRouter LLM
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

api_key = os.getenv("OPENROUTER_API_KEY")


if not api_key:

    print("ERROR: OPENROUTER_API_KEY not found.")

    print("Check your .env file.")

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

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 4. JEWELLERY PRODUCT DATABASE
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
            "A classic 22K gold ring suitable for everyday wear."
    },

    {
        "id": "R002",
        "name": "Elegant 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 22,
        "price": 25000,
        "description":
            "An elegant 22K gold ring with a traditional design."
    },

    {
        "id": "R003",
        "name": "Diamond Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 18,
        "price": 19500,
        "description":
            "An 18K gold diamond ring designed for special occasions."
    },

    {
        "id": "R004",
        "name": "Silver Ring",
        "category": "ring",
        "metal": "silver",
        "karat": 0,
        "price": 8000,
        "description":
            "A stylish silver ring at an affordable price."
    },

    {
        "id": "N001",
        "name": "22K Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": 22,
        "price": 35000,
        "description":
            "A beautiful 22K gold necklace suitable for festive occasions."
    },

    {
        "id": "N002",
        "name": "Diamond Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": 18,
        "price": 45000,
        "description":
            "An 18K gold necklace featuring diamond detailing."
    },

    {
        "id": "B001",
        "name": "22K Gold Bracelet",
        "category": "bracelet",
        "metal": "gold",
        "karat": 22,
        "price": 22000,
        "description":
            "A 22K gold bracelet with a simple traditional design."
    },

    {
        "id": "E001",
        "name": "Gold Earrings",
        "category": "earrings",
        "metal": "gold",
        "karat": 22,
        "price": 15000,
        "description":
            "22K gold earrings suitable for everyday and festive wear."
    }
]


# ============================================================
# 5. CREATE SEARCH DOCUMENTS
# ============================================================

search_documents = []


for product in products:

    document = f"""
Product ID: {product['id']}
Product Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}
Price: ₹{product['price']}

Description:
{product['description']}
"""

    search_documents.append(document)


# ============================================================
# 6. CREATE PRODUCT EMBEDDINGS
# ============================================================

print("\nCreating product embeddings...")

product_embeddings = embedding_model.encode(
    search_documents
)

print("Product embeddings created.")


# ============================================================
# 7. PRICE PARSER
# ============================================================

def parse_price(value, unit=None):

    value = value.replace(",", "")

    number = float(value)

    if unit and unit.lower() == "k":

        number *= 1000

    return int(number)


# ============================================================
# 8. EXTRACT METADATA FILTERS
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

    if re.search(r"\brings?\b", query):

        filters["category"] = "ring"

    elif re.search(r"\bnecklaces?\b", query):

        filters["category"] = "necklace"

    elif re.search(r"\bbracelets?\b", query):

        filters["category"] = "bracelet"

    elif re.search(
        r"\b(?:earrings?|ear\s*rings?)\b",
        query
    ):

        filters["category"] = "earrings"


    # ========================================================
    # METAL
    # ========================================================

    if re.search(r"\bgold\b", query):

        filters["metal"] = "gold"

    elif re.search(r"\bsilver\b", query):

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
# 9. CHECK WHETHER QUERY HAS METADATA FILTERS
# ============================================================

def has_metadata_filters(filters):

    return any(
        value is not None
        for value in filters.values()
    )


# ============================================================
# 10. FILTER PRODUCTS USING METADATA
# ============================================================

def filter_products(filters):

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


        results.append(product)


    return results


# ============================================================
# 11. SEMANTIC SEARCH
# ============================================================

def semantic_search(query, k=5):

    query_embedding = embedding_model.encode(
        query
    )


    similarities = cosine_similarity(
        [query_embedding],
        product_embeddings
    )[0]


    top_indices = similarities.argsort()[::-1][:k]


    results = []


    for index in top_indices:

        results.append({

            "product": products[index],

            "score": similarities[index]
        })


    return results


# ============================================================
# 12. HYBRID SEARCH
# ============================================================

def hybrid_search(query):

    filters = extract_filters(query)


    # ========================================================
    # CASE 1:
    # QUERY CONTAINS STRUCTURED FILTERS
    # ========================================================

    if has_metadata_filters(filters):

        metadata_results = filter_products(
            filters
        )


        # ----------------------------------------------------
        # If products match metadata,
        # use them as the main results.
        # ----------------------------------------------------

        if metadata_results:

            return [

                {
                    "product": product,

                    "score": None,

                    "source": "metadata"
                }

                for product in metadata_results
            ]


        # ----------------------------------------------------
        # No exact metadata match
        # ----------------------------------------------------

        return []


    # ========================================================
    # CASE 2:
    # NO STRUCTURED FILTERS
    #
    # Use semantic search.
    # ========================================================

    semantic_results = semantic_search(
        query,
        k=5
    )


    return [

        {
            "product": item["product"],

            "score": item["score"],

            "source": "semantic"
        }

        for item in semantic_results
    ]


# ============================================================
# 13. BUILD LLM CONTEXT
# ============================================================

def build_context(results):

    context_parts = []


    for item in results:

        product = item["product"]


        context_parts.append(

            f"""
Product ID: {product['id']}
Product Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}K
Price: ₹{product['price']:,}
Description: {product['description']}
"""
        )


    return "\n".join(
        context_parts
    )


# ============================================================
# 14. GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    results
):

    if not results:

        return (
            "I couldn't find a matching product "
            "in the available catalogue."
        )


    context = build_context(
        results
    )


    prompt = f"""
You are a jewellery store assistant.

Answer the customer's question using ONLY
the product information provided below.

Do not invent:

- prices
- karat
- metal
- product names
- availability
- features

If the information is not available,
say that you don't have that information.

Keep the answer short and helpful.

PRODUCT INFORMATION:

{context}

CUSTOMER QUESTION:

{query}

ANSWER:
"""


    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[

            {
                "role": "system",
                "content":
                    "You are a jewellery shopping assistant."
            },

            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    return response.choices[0].message.content


# ============================================================
# 15. DISPLAY SEARCH RESULTS
# ============================================================

def display_results(results):

    print(
        "\n===== SEARCH RESULTS ====="
    )


    for rank, item in enumerate(
        results,
        start=1
    ):

        product = item["product"]


        if item["score"] is not None:

            print(
                f"\nRank {rank}"
                f" | Semantic Score = "
                f"{item['score']:.3f}"
            )

        else:

            print(
                f"\nRank {rank}"
                f" | Exact Metadata Match"
            )


        print(
            f"{product['id']} - "
            f"{product['name']}"
        )

        print(
            f"Metal: {product['metal']}"
        )

        print(
            f"Karat: {product['karat']}K"
        )

        print(
            f"Price: ₹{product['price']:,}"
        )


# ============================================================
# 16. MAIN CHAT LOOP
# ============================================================

print("\n========================================")
print("      HYBRID JEWELLERY RAG")
print("========================================")

print(
    "\nExamples:"
)

print(
    "show me 22k gold rings under 20k"
)

print(
    "show me gold necklaces above 30k"
)

print(
    "I want something for a wedding"
)

print(
    "tell me about the diamond ring"
)

print(
    "\nType 'exit' to quit."
)


while True:


    # ========================================================
    # USER QUERY
    # ========================================================

    query = input(
        "\nAsk a question: "
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
    # EXTRACT FILTERS
    # ========================================================

    filters = extract_filters(
        query
    )


    print(
        "\n===== DETECTED FILTERS ====="
    )


    for key, value in filters.items():

        print(
            f"{key:12} → {value}"
        )


    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    results = hybrid_search(
        query
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    display_results(
        results
    )


    # ========================================================
    # GENERATE ANSWER
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


        print(answer)


    except Exception as e:

        print(
            "\nERROR WHILE GENERATING ANSWER:"
        )

        print(e)




"""
Question
   ↓
Query Understanding
   ↓
┌──────────────────────┐
│ Has metadata filters?│
└──────────┬───────────┘
           │
       YES │ NO
           │
     ┌─────┘ └──────────┐
     ↓                  ↓
Metadata             Vector
Filtering            Search
     ↓                  ↓
Exact Products      Relevant Products
     └────────┬─────────┘
              ↓
             LLM
              ↓
        Final Answer
"""