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
    print("Please check your .env file.")
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

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 4. PRODUCT DATABASE
# ============================================================

products = [

    {
        "id": "R001",
        "name": "Classic 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 18000,
        "description":
            "A classic 22K gold ring suitable for everyday wear."
    },

    {
        "id": "R002",
        "name": "Elegant 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 25000,
        "description":
            "An elegant 22K gold ring with a beautiful traditional design, "
            "suitable for weddings and special occasions."
    },

    {
        "id": "R003",
        "name": "Traditional 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 27000,
        "description":
            "A traditional 22K gold ring with an elegant traditional design."
    },

    {
        "id": "R004",
        "name": "Bridal 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 29000,
        "description":
            "A bridal 22K gold ring designed for weddings and bridal occasions."
    },

    {
        "id": "R005",
        "name": "Modern 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 24000,
        "description":
            "A modern 22K gold ring with a minimal contemporary design "
            "suitable for parties and special occasions."
    },

    {
        "id": "R006",
        "name": "Diamond Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "18K",
        "price": 19500,
        "description":
            "An 18K gold diamond ring with a modern design, "
            "ideal for special occasions and weddings."
    },

    {
        "id": "N001",
        "name": "22K Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": "22K",
        "price": 35000,
        "description":
            "A beautiful 22K gold necklace suitable for festive occasions."
    },

    {
        "id": "B001",
        "name": "22K Gold Bracelet",
        "category": "bracelet",
        "metal": "gold",
        "karat": "22K",
        "price": 22000,
        "description":
            "A 22K gold bracelet with a traditional design "
            "suitable for festive occasions."
    },

    {
        "id": "E001",
        "name": "Gold Earrings",
        "category": "earrings",
        "metal": "gold",
        "karat": "22K",
        "price": 15000,
        "description":
            "22K gold earrings with an elegant design "
            "suitable for weddings and festive occasions."
    },

    {
        "id": "S001",
        "name": "Silver Ring",
        "category": "ring",
        "metal": "silver",
        "karat": "0K",
        "price": 8000,
        "description":
            "A simple silver ring suitable for everyday wear."
    }
]


# ============================================================
# 5. CREATE PRODUCT TEXT
# ============================================================

product_texts = [

    (
        f"{product['name']}. "
        f"{product['category']}. "
        f"{product['metal']}. "
        f"{product['karat']}. "
        f"{product['description']}"
    )

    for product in products
]


# ============================================================
# 6. CREATE EMBEDDINGS
# ============================================================

print("Creating product embeddings...")

product_embeddings = embedding_model.encode(
    product_texts
)

print("Product embeddings created.")


# ============================================================
# 7. PRICE EXTRACTION
# ============================================================

def extract_price(text):

    text = text.lower()

    clean_text = text.replace(",", "")


    # --------------------------------------------------------
    # Handle 30k / 30 k
    # --------------------------------------------------------

    match = re.search(
        r"(?:₹|\brs\.?\s*)?(\d+(?:\.\d+)?)\s*k\b",
        clean_text
    )


    if match:

        value = float(match.group(1))

        return int(value * 1000)


    # --------------------------------------------------------
    # Handle 30000
    # --------------------------------------------------------

    match = re.search(
        r"(?:₹|\brs\.?\s*)?(\d{4,6})\b",
        clean_text
    )


    if match:

        return int(match.group(1))


    return None


# ============================================================
# 8. FILTER EXTRACTION
# ============================================================

def extract_filters(query):

    text = query.lower()


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

    if "ring" in text:

        filters["category"] = "ring"

    elif "necklace" in text:

        filters["category"] = "necklace"

    elif "bracelet" in text:

        filters["category"] = "bracelet"

    elif "earring" in text:

        filters["category"] = "earrings"


    # ========================================================
    # METAL
    # ========================================================

    if "gold" in text:

        filters["metal"] = "gold"

    elif "silver" in text:

        filters["metal"] = "silver"


    # ========================================================
    # KARAT
    # ========================================================

    karat_match = re.search(
        r"\b(18|20|22|24)\s*k\b",
        text
    )


    if karat_match:

        filters["karat"] = (
            karat_match.group(1) + "K"
        )


    # ========================================================
    # PRICE
    # ========================================================

    price = extract_price(text)


    # ========================================================
    # MAX PRICE
    # ========================================================

    if any(
        phrase in text
        for phrase in [
            "under",
            "below",
            "less than",
            "within",
            "upto",
            "up to"
        ]
    ):

        if price is not None:

            filters["max_price"] = price


    # ========================================================
    # MIN PRICE
    # ========================================================

    if any(
        phrase in text
        for phrase in [
            "above",
            "over",
            "more than",
            "greater than"
        ]
    ):

        if price is not None:

            filters["min_price"] = price


    return filters


# ============================================================
# 9. DISPLAY FILTERS
# ============================================================

def display_filters(filters):

    print("\n===== DETECTED FILTERS =====")

    print(
        f"category    -> {filters['category']}"
    )

    print(
        f"metal       -> {filters['metal']}"
    )

    print(
        f"karat       -> {filters['karat']}"
    )

    print(
        f"max_price   -> {filters['max_price']}"
    )

    print(
        f"min_price   -> {filters['min_price']}"
    )


# ============================================================
# 10. HARD METADATA FILTER
# ============================================================

def metadata_filter(products, filters):

    results = []


    for product in products:


        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        if filters["category"]:

            if product["category"] != filters["category"]:

                continue


        # ----------------------------------------------------
        # METAL
        # ----------------------------------------------------

        if filters["metal"]:

            if product["metal"] != filters["metal"]:

                continue


        # ----------------------------------------------------
        # KARAT
        # ----------------------------------------------------

        if filters["karat"]:

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

def semantic_search(
    query,
    filtered_products,
    top_k=3
):

    if not filtered_products:

        return []


    # --------------------------------------------------------
    # Find original indexes
    # --------------------------------------------------------

    filtered_indexes = [

        products.index(product)

        for product in filtered_products

    ]


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query
    )


    # --------------------------------------------------------
    # Filtered embeddings
    # --------------------------------------------------------

    filtered_embeddings = product_embeddings[
        filtered_indexes
    ]


    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        [query_embedding],
        filtered_embeddings
    )[0]


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranked_indexes = similarities.argsort()[::-1]


    results = []


    for index in ranked_indexes[:top_k]:

        results.append({

            "product": filtered_products[index],

            "score": float(
                similarities[index]
            )

        })


    return results


# ============================================================
# 12. DISPLAY PRODUCTS
# ============================================================

def display_products(results):

    print(
        "\n===== FINAL RETRIEVED PRODUCTS ====="
    )


    for rank, result in enumerate(
        results,
        start=1
    ):

        product = result["product"]

        score = result["score"]


        print(f"\nRank {rank}")

        print(
            f"Semantic Score = {score:.3f}"
        )

        print(
            f"ID: {product['id']}"
        )

        print(
            f"Name: {product['name']}"
        )

        print(
            f"Category: {product['category']}"
        )

        print(
            f"Metal: {product['metal']}"
        )

        print(
            f"Karat: {product['karat']}"
        )

        print(
            f"Price: ₹{product['price']:,}"
        )

        print(
            f"Description: {product['description']}"
        )


# ============================================================
# 13. CREATE STRICT GROUNDED CONTEXT
# ============================================================

def create_context(results):

    context = []


    for result in results:

        product = result["product"]


        context.append(

            f"""
PRODUCT ID: {product['id']}
PRODUCT NAME: {product['name']}
CATEGORY: {product['category']}
METAL: {product['metal']}
KARAT: {product['karat']}
PRICE: ₹{product['price']:,}
DESCRIPTION: {product['description']}
"""

        )


    return "\n".join(context)


# ============================================================
# 14. GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(
    query,
    results
):

    if not results:

        return (
            "I couldn't find any products "
            "matching your requirements."
        )


    context = create_context(results)


    # ========================================================
    # STRICT GROUNDED PROMPT
    # ========================================================

    prompt = f"""
You are a jewellery shopping assistant.

Your job is to answer the customer's question
using ONLY the product data in the context.

STRICT RULES:

1. Do not invent products.

2. Do not invent prices.

3. Do not invent karat information.

4. Do not invent product features.

5. Do not change any product information.

6. Do not reinterpret or correct the customer's
   wording unless it is absolutely necessary.

7. Do not make assumptions about products.

8. If the customer's exact requested product
   characteristic is not present in the context,
   clearly say that the available product data
   does not confirm it.

9. If products are provided, recommend ONLY
   products from the context.

10. When mentioning a price, use exactly the
    price provided in the context.

11. When the customer asks for products under
    a certain price, do not recommend anything
    above that price.

12. Keep the answer concise and natural.

13. Never mention internal processes such as:
    embeddings, vector search, semantic search,
    metadata filtering, prompts, or retrieval.

CUSTOMER QUESTION:
{query}

PRODUCT CONTEXT:
{context}

Now answer the customer.
"""


    # ========================================================
    # OPENROUTER
    # ========================================================

    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[

            {
                "role": "system",
                "content":
                    "You are a strictly grounded jewellery "
                    "shopping assistant."
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )


    return response.choices[0].message.content


# ============================================================
# 15. MAIN CHATBOT LOOP
# ============================================================

print("\n==============================================")

print(
    "          GROUNDED JEWELLERY RAG"
)

print("==============================================")


print("\nExamples:")

print(
    "Show me gold rings under 30k"
)

print(
    "I want a traditional gold ring for wedding"
)

print(
    "Show me a 22K gold ring"
)

print(
    "I want something under 20k"
)

print(
    "\nType 'exit' to quit."
)


while True:


    # ========================================================
    # USER INPUT
    # ========================================================

    query = input(
        "\nAsk your jewellery question: "
    ).strip()


    # ========================================================
    # EXIT
    # ========================================================

    if query.lower() == "exit":

        print("\nGoodbye!")

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
    # FILTER EXTRACTION
    # ========================================================

    filters = extract_filters(query)


    display_filters(filters)


    # ========================================================
    # HARD FILTER
    # ========================================================

    filtered_products = metadata_filter(
        products,
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
            f"- {product['name']} | "
            f"₹{product['price']:,}"
        )


    # ========================================================
    # NO MATCH
    # ========================================================

    if not filtered_products:

        print(
            "\n===== FINAL ANSWER ====="
        )

        print(
            "I couldn't find any products "
            "matching your requirements."
        )

        continue


    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    results = semantic_search(

        query,

        filtered_products,

        top_k=3

    )


    # ========================================================
    # DISPLAY RETRIEVED PRODUCTS
    # ========================================================

    display_products(results)


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
            "\nERROR while generating answer:"
        )

        print(e)