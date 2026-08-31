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
# 5. CREATE PRODUCT TEXT FOR EMBEDDINGS
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
# 6. CREATE PRODUCT EMBEDDINGS
# ============================================================

print("Creating product embeddings...")

product_embeddings = embedding_model.encode(
    product_texts
)

print("Product embeddings created.")


# ============================================================
# 7. EXTRACT PRICE
# ============================================================

def extract_price(text):

    text = text.lower()

    # Remove commas
    clean_text = text.replace(",", "")

    # Examples:
    # 30000
    # 30k
    # 30 k
    # ₹30000
    # ₹30k

    match = re.search(
        r"(?:₹|\brs\.?\s*)?(\d+(?:\.\d+)?)\s*k\b",
        clean_text
    )

    if match:

        value = float(match.group(1))

        return int(value * 1000)


    match = re.search(
        r"(?:₹|\brs\.?\s*)?(\d{4,6})\b",
        clean_text
    )

    if match:

        return int(match.group(1))


    return None


# ============================================================
# 8. EXTRACT FILTERS
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


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    categories = [

        "ring",
        "rings",
        "necklace",
        "necklaces",
        "bracelet",
        "bracelets",
        "earring",
        "earrings"

    ]


    for category in categories:

        if category in text:

            if category in ["ring", "rings"]:
                filters["category"] = "ring"

            elif category in ["necklace", "necklaces"]:
                filters["category"] = "necklace"

            elif category in ["bracelet", "bracelets"]:
                filters["category"] = "bracelet"

            elif category in ["earring", "earrings"]:
                filters["category"] = "earrings"

            break


    # --------------------------------------------------------
    # METAL
    # --------------------------------------------------------

    if "gold" in text:

        filters["metal"] = "gold"

    elif "silver" in text:

        filters["metal"] = "silver"


    # --------------------------------------------------------
    # KARAT
    # --------------------------------------------------------

    karat_match = re.search(
        r"\b(18|20|22|24)\s*k\b",
        text
    )

    if karat_match:

        filters["karat"] = karat_match.group(1) + "K"


    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    price = extract_price(text)


    # --------------------------------------------------------
    # MAX PRICE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # MIN PRICE
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # PRODUCT PASSED ALL HARD FILTERS
        # ----------------------------------------------------

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
    # Get indexes of filtered products
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
    # Only compare against filtered products
    # --------------------------------------------------------

    filtered_embeddings = product_embeddings[
        filtered_indexes
    ]


    similarities = cosine_similarity(
        [query_embedding],
        filtered_embeddings
    )[0]


    # --------------------------------------------------------
    # Sort by semantic similarity
    # --------------------------------------------------------

    ranked_indexes = similarities.argsort()[::-1]


    results = []


    for index in ranked_indexes[:top_k]:

        product = filtered_products[index]

        score = similarities[index]


        results.append({

            "product": product,

            "score": float(score)

        })


    return results


# ============================================================
# 12. DETECT EXHAUSTIVE REQUEST
# ============================================================

def is_exhaustive_request(query):

    text = query.lower()

    phrases = [

        "show me all",
        "show all",
        "all products",
        "everything",
        "list all",
        "list everything",
        "what are all",
        "give me all"

    ]


    for phrase in phrases:

        if phrase in text:

            return True


    return False


# ============================================================
# 13. DISPLAY MATCHING PRODUCTS
# ============================================================

def display_metadata_results(filtered_products):

    print("\n===== METADATA FILTER RESULTS =====")

    print(
        f"Matching products: {len(filtered_products)}"
    )


    for product in filtered_products:

        print(
            f"- {product['name']} | "
            f"₹{product['price']:,}"
        )


# ============================================================
# 14. DISPLAY SEMANTIC RESULTS
# ============================================================

def display_semantic_results(results):

    print("\n===== SEMANTICALLY RANKED PRODUCTS =====")


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
            f"Price: ₹{product['price']:,}"
        )

        print(
            f"Karat: {product['karat']}"
        )

        print(
            f"Description: {product['description']}"
        )


# ============================================================
# 15. GENERATE FINAL ANSWER
# ============================================================

def generate_answer(
    query,
    results
):

    if not results:

        return (
            "I couldn't find any products matching "
            "your requirements."
        )


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []


    for result in results:

        product = result["product"]


        context_parts.append(

            f"""
Product ID: {product['id']}
Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}
Price: ₹{product['price']:,}
Description: {product['description']}
"""

        )


    context = "\n".join(context_parts)


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a jewellery shopping assistant.

Answer the customer's question using ONLY
the product information provided below.

Do not invent products.

Do not invent prices.

Do not change prices.

Do not claim that a product exists if it is
not present in the provided context.

If the customer asks for products, clearly
mention the product name and price.

If multiple products are provided, you may
recommend the most relevant ones.

If no product satisfies the request, say that
no matching product was found.

Customer question:
{query}

Available product information:
{context}

Give a helpful and concise answer.
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
                    "You are a jewellery shopping assistant. "
                    "Use only the supplied product data."
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )


    return response.choices[0].message.content


# ============================================================
# 16. MAIN CHAT LOOP
# ============================================================

print("\n==============================================")

print(
    "       JEWELLERY RAG CHATBOT - PROJECT 21"
)

print("==============================================")


print("\nYou can ask things like:")

print(
    "- Show me gold rings under 30k"
)

print(
    "- I want a traditional gold ring for wedding"
)

print(
    "- Show me all gold products under 30k"
)

print(
    "- I want a 22K gold ring"
)

print(
    "- Show me something above 20k"
)

print(
    "\nType 'exit' to quit."
)


while True:


    # ========================================================
    # GET USER QUERY
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
    # EMPTY INPUT
    # ========================================================

    if not query:

        print(
            "Please enter a question."
        )

        continue


    # ========================================================
    # STEP 1 - EXTRACT FILTERS
    # ========================================================

    filters = extract_filters(query)


    display_filters(filters)


    # ========================================================
    # STEP 2 - HARD METADATA FILTER
    # ========================================================

    filtered_products = metadata_filter(
        products,
        filters
    )


    display_metadata_results(
        filtered_products
    )


    # ========================================================
    # NO PRODUCTS
    # ========================================================

    if not filtered_products:

        print("\n===== FINAL ANSWER =====")

        print(
            "I couldn't find any products "
            "matching your requirements."
        )

        continue


    # ========================================================
    # STEP 3 - EXHAUSTIVE REQUEST
    # ========================================================

    if is_exhaustive_request(query):


        print(
            "\n===== EXHAUSTIVE REQUEST ====="
        )

        print(
            "Returning all products that satisfy "
            "the hard filters."
        )


        final_results = [

            {
                "product": product,
                "score": 1.0
            }

            for product in filtered_products

        ]


    else:


        # ====================================================
        # STEP 4 - SEMANTIC SEARCH
        # ====================================================

        final_results = semantic_search(

            query,

            filtered_products,

            top_k=3

        )


        display_semantic_results(
            final_results
        )


    # ========================================================
    # STEP 5 - GENERATE ANSWER
    # ========================================================

    print(
        "\n===== GENERATING ANSWER ====="
    )


    try:


        answer = generate_answer(

            query,

            final_results

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