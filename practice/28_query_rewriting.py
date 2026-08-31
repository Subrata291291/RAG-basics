import os
import json

from dotenv import load_dotenv
from openai import OpenAI


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
# 4. PRODUCT CATALOG
# ============================================================

products = [

    {
        "id": "R001",
        "name": "Classic 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 18000,
        "description": "A classic 22K gold ring suitable for everyday wear."
    },

    {
        "id": "R002",
        "name": "Elegant 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 25000,
        "description": (
            "An elegant 22K gold ring with a beautiful traditional "
            "design, suitable for weddings and special occasions."
        )
    },

    {
        "id": "R003",
        "name": "Traditional 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 27000,
        "description": (
            "A traditional 22K gold ring suitable for weddings "
            "and festive occasions."
        )
    },

    {
        "id": "R004",
        "name": "Bridal 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 29000,
        "description": (
            "A bridal 22K gold ring with an ornate design "
            "for weddings."
        )
    },

    {
        "id": "R005",
        "name": "Modern 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 24000,
        "description": (
            "A modern 22K gold ring with a minimal contemporary "
            "design suitable for parties and special occasions."
        )
    },

    {
        "id": "R006",
        "name": "Diamond Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "18K",
        "price": 19500,
        "description": (
            "An 18K gold diamond ring with a modern design, "
            "ideal for special occasions and weddings."
        )
    },

    {
        "id": "S001",
        "name": "Silver Ring",
        "category": "ring",
        "metal": "silver",
        "karat": "0K",
        "price": 8000,
        "description": (
            "A simple silver ring suitable for everyday wear."
        )
    },

    {
        "id": "N001",
        "name": "22K Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": "22K",
        "price": 35000,
        "description": (
            "A traditional 22K gold necklace suitable for "
            "festive occasions."
        )
    },

    {
        "id": "N002",
        "name": "Diamond Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": "18K",
        "price": 45000,
        "description": (
            "An elegant 18K diamond gold necklace for "
            "special occasions."
        )
    },

    {
        "id": "B001",
        "name": "22K Gold Bracelet",
        "category": "bracelet",
        "metal": "gold",
        "karat": "22K",
        "price": 22000,
        "description": (
            "A 22K gold bracelet with a traditional design "
            "suitable for festive occasions."
        )
    },

    {
        "id": "E001",
        "name": "Gold Earrings",
        "category": "earrings",
        "metal": "gold",
        "karat": "22K",
        "price": 15000,
        "description": (
            "22K gold earrings with an elegant design suitable "
            "for weddings and festive occasions."
        )
    }
]


# ============================================================
# 5. CONVERSATION STATE
# ============================================================

conversation = []


# ============================================================
# 6. DISPLAY PRODUCTS
# ============================================================

def display_products():

    print("\n===== AVAILABLE PRODUCTS =====")

    for product in products:

        print(
            f"{product['id']} | "
            f"{product['name']} | "
            f"{product['metal']} | "
            f"{product['karat']} | "
            f"₹{product['price']:,}"
        )


# ============================================================
# 7. QUERY REWRITING
# ============================================================

def rewrite_query(user_query, conversation_history):

    history_text = ""

    for message in conversation_history:

        history_text += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )


    prompt = f"""
You are a query rewriting system for a jewellery shopping chatbot.

Your job is to convert the user's latest message into a
complete standalone search query.

Use the previous conversation when the latest message is
a follow-up.

Examples:

Previous:
User: show me gold rings under 30000

Latest:
only 22K

Rewritten:
show me 22K gold rings under 30000


Previous:
User: show me gold rings under 30000
Assistant: Here are the matching products.

Latest:
which one is cheapest?

Rewritten:
which gold ring under 30000 is cheapest?


Previous:
User: show me gold rings under 30000
User: only 22K

Latest:
what about necklaces?

Rewritten:
show me 22K gold necklaces under 30000


IMPORTANT:

- Preserve previous filters when the user is clearly
  continuing the same search.
- If the user changes a filter, update it.
- If the user starts a completely new request, do not
  carry unrelated old filters.
- Do not invent product information.
- Do not answer the question.
- Return ONLY the rewritten query.

Conversation:
{history_text}

Latest user message:
{user_query}

Standalone rewritten query:
"""


    try:

        response = client.chat.completions.create(

            model="openrouter/free",

            messages=[

                {
                    "role": "system",
                    "content": (
                        "You rewrite shopping queries. "
                        "Return only the rewritten query."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ]
        )


        rewritten = response.choices[0].message.content.strip()

        return rewritten


    except Exception as e:

        print("\nERROR during query rewriting:")
        print(e)

        return user_query


# ============================================================
# 8. SIMPLE FILTER EXTRACTION
# ============================================================

def extract_filters(query):

    query_lower = query.lower()

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

        if category in query_lower:

            if category.startswith("ring"):
                filters["category"] = "ring"

            elif category.startswith("necklace"):
                filters["category"] = "necklace"

            elif category.startswith("bracelet"):
                filters["category"] = "bracelet"

            elif category.startswith("earring"):
                filters["category"] = "earrings"

            break


    # --------------------------------------------------------
    # METAL
    # --------------------------------------------------------

    if "gold" in query_lower:

        filters["metal"] = "gold"

    elif "silver" in query_lower:

        filters["metal"] = "silver"


    # --------------------------------------------------------
    # KARAT
    # --------------------------------------------------------

    for karat in ["18K", "22K", "24K"]:

        if karat.lower() in query_lower:

            filters["karat"] = karat

            break


    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    import re


    numbers = re.findall(
        r"\d[\d,]*",
        query
    )


    numbers = [

        int(number.replace(",", ""))

        for number in numbers

    ]


    # --------------------------------------------------------
    # UNDER / BELOW / LESS THAN
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "under",
            "below",
            "less than",
            "upto",
            "up to"
        ]
    ):

        if numbers:

            filters["max_price"] = numbers[-1]


    # --------------------------------------------------------
    # ABOVE / OVER / MORE THAN
    # --------------------------------------------------------

    elif any(
        word in query_lower
        for word in [
            "above",
            "over",
            "more than"
        ]
    ):

        if numbers:

            filters["min_price"] = numbers[-1]


    return filters


# ============================================================
# 9. FILTER PRODUCTS
# ============================================================

def filter_products(filters):

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
# 10. DISPLAY FILTERS
# ============================================================

def display_filters(filters):

    print("\n===== EXTRACTED FILTERS =====")

    print(
        f"category -> {filters['category']}"
    )

    print(
        f"metal -> {filters['metal']}"
    )

    print(
        f"karat -> {filters['karat']}"
    )

    print(
        f"min_price -> {filters['min_price']}"
    )

    print(
        f"max_price -> {filters['max_price']}"
    )


# ============================================================
# 11. DISPLAY SEARCH RESULTS
# ============================================================

def display_results(results):

    print("\n===== SEARCH RESULTS =====")

    if not results:

        print("No matching products found.")

        return


    for index, product in enumerate(
        results,
        start=1
    ):

        print(f"\nRank {index}")

        print(
            f"ID: {product['id']}"
        )

        print(
            f"Name: {product['name']}"
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
# 12. GENERATE PRODUCT ANSWER
# ============================================================

def generate_product_answer(
    rewritten_query,
    results
):

    if not results:

        return (
            "I couldn't find any products matching "
            "your requirements."
        )


    # --------------------------------------------------------
    # Create trusted catalog context
    # --------------------------------------------------------

    context = ""

    for product in results:

        context += f"""
Product ID: {product['id']}
Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}
Price: ₹{product['price']:,}
Description: {product['description']}
"""


    prompt = f"""
You are a jewellery shopping assistant.

Answer the user's request using ONLY the products
provided below.

Do not invent products.
Do not invent prices.
Do not invent karats.
Do not invent features.

If multiple products match, list the matching products.

Catalog:

{context}

User's rewritten query:

{rewritten_query}

Give a concise helpful answer.
"""


    try:

        response = client.chat.completions.create(

            model="openrouter/free",

            messages=[

                {
                    "role": "system",
                    "content": (
                        "You answer jewellery shopping "
                        "questions using catalog data only."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ]
        )


        return response.choices[0].message.content.strip()


    except Exception as e:

        print("\nERROR during answer generation:")

        print(e)

        # ----------------------------------------------------
        # Safe fallback
        # ----------------------------------------------------

        answer = "Here are the matching products:\n"

        for product in results:

            answer += (
                f"- {product['name']} "
                f"({product['karat']}) - "
                f"₹{product['price']:,}\n"
            )


        return answer


# ============================================================
# 13. MAIN CHAT LOOP
# ============================================================

print("\n==========================================")
print("       QUERY REWRITING RAG")
print("==========================================")

print("\nJewellery conversational search")

print("Type 'exit' to quit.")


while True:

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    user_query = input(
        "\nAsk your jewellery question: "
    ).strip()


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if user_query.lower() == "exit":

        print("\nGoodbye!")

        break


    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not user_query:

        print("Please enter a question.")

        continue


    # ========================================================
    # QUERY REWRITING
    # ========================================================

    print("\n===== QUERY REWRITING =====")

    rewritten_query = rewrite_query(
        user_query,
        conversation
    )


    print(
        f"Original query:  {user_query}"
    )

    print(
        f"Rewritten query: {rewritten_query}"
    )


    # ========================================================
    # FILTER EXTRACTION
    # ========================================================

    filters = extract_filters(
        rewritten_query
    )


    display_filters(
        filters
    )


    # ========================================================
    # PRODUCT FILTERING
    # ========================================================

    results = filter_products(
        filters
    )


    print("\n===== MATCHING PRODUCTS =====")

    print(
        f"Found {len(results)} product(s)."
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

    print("\n===== GENERATING ANSWER =====")

    answer = generate_product_answer(
        rewritten_query,
        results
    )


    print("\n===== FINAL ANSWER =====")

    print(answer)


    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    conversation.append({

        "role": "user",

        "content": user_query

    })


    conversation.append({

        "role": "assistant",

        "content": answer

    })