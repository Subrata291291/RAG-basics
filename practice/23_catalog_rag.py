import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


if not api_key:

    print("ERROR: OPENROUTER_API_KEY not found.")

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
# 4. LOAD PRODUCTS FROM JSON
# ============================================================

PRODUCT_FILE = "data/products.json"


with open(
    PRODUCT_FILE,
    "r",
    encoding="utf-8"
) as file:

    products = json.load(file)


print(
    f"Loaded {len(products)} products."
)


# ============================================================
# 5. CREATE PRODUCT TEXT
# ============================================================

product_texts = []


for product in products:

    text = f"""
    Product name: {product['name']}
    Category: {product['category']}
    Metal: {product['metal']}
    Karat: {product['karat']}
    Description: {product['description']}
    """

    product_texts.append(text)


# ============================================================
# 6. CREATE PRODUCT EMBEDDINGS
# ============================================================

print("Creating product embeddings...")

product_embeddings = embedding_model.encode(
    product_texts
)

print("Product embeddings created.")


# ============================================================
# 7. LOAD KNOWLEDGE DOCUMENTS
# ============================================================

KNOWLEDGE_FOLDER = "data/knowledge"


knowledge_documents = []


for filename in os.listdir(KNOWLEDGE_FOLDER):

    if not filename.endswith(".txt"):

        continue


    path = os.path.join(
        KNOWLEDGE_FOLDER,
        filename
    )


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()


    knowledge_documents.append({

        "name": filename,

        "content": content

    })


print(
    f"Loaded {len(knowledge_documents)} knowledge documents."
)


# ============================================================
# 8. CREATE KNOWLEDGE EMBEDDINGS
# ============================================================

knowledge_texts = [

    document["content"]

    for document in knowledge_documents

]


knowledge_embeddings = embedding_model.encode(
    knowledge_texts
)


# ============================================================
# 9. PRICE EXTRACTION
# ============================================================

def extract_price(text):

    text = text.lower()

    text = text.replace(",", "")


    # --------------------------------------------------------
    # 30k
    # --------------------------------------------------------

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*k\b",
        text
    )


    if match:

        return int(
            float(match.group(1)) * 1000
        )


    # --------------------------------------------------------
    # 30000
    # --------------------------------------------------------

    match = re.search(
        r"\b(\d{4,6})\b",
        text
    )


    if match:

        return int(match.group(1))


    return None


# ============================================================
# 10. EXTRACT PRODUCT FILTERS
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

    match = re.search(
        r"\b(18|20|22|24)\s*k\b",
        text
    )


    if match:

        filters["karat"] = (
            match.group(1) + "K"
        )


    # ========================================================
    # PRICE
    # ========================================================

    price = extract_price(text)


    if any(
        phrase in text
        for phrase in [
            "under",
            "below",
            "less than",
            "upto",
            "up to"
        ]
    ):

        if price:

            filters["max_price"] = price


    if any(
        phrase in text
        for phrase in [
            "above",
            "over",
            "more than",
            "greater than"
        ]
    ):

        if price:

            filters["min_price"] = price


    return filters


# ============================================================
# 11. METADATA FILTER
# ============================================================

def filter_products(
    products,
    filters
):

    results = []


    for product in products:


        if filters["category"]:

            if product["category"] != filters["category"]:

                continue


        if filters["metal"]:

            if product["metal"] != filters["metal"]:

                continue


        if filters["karat"]:

            if product["karat"] != filters["karat"]:

                continue


        if filters["max_price"] is not None:

            if product["price"] > filters["max_price"]:

                continue


        if filters["min_price"] is not None:

            if product["price"] < filters["min_price"]:

                continue


        results.append(product)


    return results


# ============================================================
# 12. PRODUCT SEMANTIC SEARCH
# ============================================================

def product_search(
    query,
    filtered_products,
    top_k=3
):

    if not filtered_products:

        return []


    original_indexes = [

        products.index(product)

        for product in filtered_products

    ]


    query_embedding = embedding_model.encode(
        query
    )


    filtered_embeddings = product_embeddings[
        original_indexes
    ]


    similarities = cosine_similarity(

        [query_embedding],

        filtered_embeddings

    )[0]


    ranked_indexes = similarities.argsort()[::-1]


    results = []


    for index in ranked_indexes[:top_k]:

        results.append({

            "product":
                filtered_products[index],

            "score":
                float(similarities[index])

        })


    return results


# ============================================================
# 13. KNOWLEDGE DOCUMENT SEARCH
# ============================================================

def search_knowledge(
    query,
    top_k=2
):

    query_embedding = embedding_model.encode(
        query
    )


    similarities = cosine_similarity(

        [query_embedding],

        knowledge_embeddings

    )[0]


    ranked_indexes = similarities.argsort()[::-1]


    results = []


    for index in ranked_indexes[:top_k]:

        results.append({

            "document":
                knowledge_documents[index],

            "score":
                float(similarities[index])

        })


    return results


# ============================================================
# 14. DETECT QUERY TYPE
# ============================================================

def detect_query_type(query):

    text = query.lower()


    policy_words = [

        "privacy",

        "private",

        "personal information",

        "terms",

        "conditions",

        "shipping",

        "delivery",

        "return",

        "refund",

        "policy"

    ]


    for word in policy_words:

        if word in text:

            return "knowledge"


    product_words = [

        "ring",

        "necklace",

        "bracelet",

        "earring",

        "jewellery",

        "jewelry",

        "gold",

        "silver",

        "karat",

        "price",

        "cost",

        "buy"

    ]


    for word in product_words:

        if word in text:

            return "product"


    return "knowledge"


# ============================================================
# 15. CREATE PRODUCT CONTEXT
# ============================================================

def create_product_context(results):

    context = ""


    for result in results:

        product = result["product"]


        context += f"""

PRODUCT ID: {product['id']}

NAME: {product['name']}

CATEGORY: {product['category']}

METAL: {product['metal']}

KARAT: {product['karat']}

PRICE: ₹{product['price']:,}

DESCRIPTION:
{product['description']}

"""


    return context


# ============================================================
# 16. CREATE KNOWLEDGE CONTEXT
# ============================================================

def create_knowledge_context(results):

    context = ""


    for result in results:

        document = result["document"]


        context += f"""

DOCUMENT:
{document['name']}

CONTENT:
{document['content']}

"""


    return context


# ============================================================
# 17. GENERATE PRODUCT ANSWER
# ============================================================

def generate_product_answer(
    query,
    results
):

    context = create_product_context(
        results
    )


    prompt = f"""
You are a jewellery shopping assistant.

Answer the customer's question using ONLY
the product information provided below.

STRICT RULES:

- Never invent a product.
- Never invent a price.
- Never invent a feature.
- Never invent a karat.
- Never recommend a product outside the provided context.
- Never change product information.
- Do not make assumptions.
- If the requested product does not exist,
  clearly say that it is not available in the
  provided product catalog.
- If multiple products match, list the useful options.
- Keep the answer concise.
- Do not mention embeddings, RAG, vector search,
  metadata filtering or internal systems.

CUSTOMER QUESTION:
{query}

PRODUCT CATALOG CONTEXT:
{context}

Answer the customer.
"""


    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[

            {
                "role": "system",
                "content":
                    "You are a strictly grounded "
                    "jewellery shopping assistant."
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )


    return response.choices[0].message.content


# ============================================================
# 18. GENERATE KNOWLEDGE ANSWER
# ============================================================

def generate_knowledge_answer(
    query,
    results
):

    context = create_knowledge_context(
        results
    )


    prompt = f"""
You are a customer support assistant for a
jewellery website.

Answer the customer's question using ONLY
the information in the provided documents.

STRICT RULES:

- Do not invent policies.
- Do not invent delivery times.
- Do not invent return conditions.
- Do not invent refund information.
- Do not invent privacy practices.
- Do not use outside knowledge.
- If the answer is not present in the documents,
  say that the provided information does not
  contain the answer.
- Keep the answer clear and concise.

CUSTOMER QUESTION:
{query}

KNOWLEDGE DOCUMENTS:
{context}

Answer the customer.
"""


    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[

            {
                "role": "system",
                "content":
                    "You are a strictly grounded "
                    "customer support assistant."
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )


    return response.choices[0].message.content


# ============================================================
# 19. MAIN CHAT LOOP
# ============================================================

print()
print("=" * 50)
print("       JEWELLERY WEBSITE CHATBOT")
print("=" * 50)

print()
print("You can ask about:")
print("- Products")
print("- Prices")
print("- Gold / silver")
print("- Rings / necklaces / bracelets")
print("- Shipping")
print("- Returns")
print("- Privacy")
print("- Terms and conditions")

print()
print("Type 'exit' to quit.")


while True:


    # ========================================================
    # USER QUESTION
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
    # DETECT QUERY TYPE
    # ========================================================

    query_type = detect_query_type(
        query
    )


    print(
        f"\nQuery type: {query_type}"
    )


    # ========================================================
    # PRODUCT QUERY
    # ========================================================

    if query_type == "product":


        # ----------------------------------------------------
        # Extract filters
        # ----------------------------------------------------

        filters = extract_filters(
            query
        )


        print(
            "\n===== DETECTED FILTERS ====="
        )

        print(
            f"category  -> {filters['category']}"
        )

        print(
            f"metal     -> {filters['metal']}"
        )

        print(
            f"karat     -> {filters['karat']}"
        )

        print(
            f"max_price -> {filters['max_price']}"
        )

        print(
            f"min_price -> {filters['min_price']}"
        )


        # ----------------------------------------------------
        # Metadata filtering
        # ----------------------------------------------------

        filtered_products = filter_products(

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


        # ----------------------------------------------------
        # No products
        # ----------------------------------------------------

        if not filtered_products:

            print(
                "\n===== FINAL ANSWER ====="
            )

            print(
                "I couldn't find any products "
                "matching your requirements."
            )

            continue


        # ----------------------------------------------------
        # Semantic search
        # ----------------------------------------------------

        results = product_search(

            query,

            filtered_products,

            top_k=3

        )


        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print(
            "\n===== PRODUCT SEARCH RESULTS ====="
        )


        for rank, result in enumerate(

            results,

            start=1

        ):

            product = result["product"]

            score = result["score"]


            print(
                f"\nRank {rank}"
            )

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
                f"Description: "
                f"{product['description']}"
            )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        print(
            "\n===== GENERATING ANSWER ====="
        )


        try:

            answer = generate_product_answer(

                query,

                results

            )


            print(
                "\n===== FINAL ANSWER ====="
            )

            print(answer)


        except Exception as e:

            print(
                "\nERROR:"
            )

            print(e)


    # ========================================================
    # KNOWLEDGE QUERY
    # ========================================================

    else:


        results = search_knowledge(

            query,

            top_k=2

        )


        print(
            "\n===== KNOWLEDGE SEARCH ====="
        )


        for rank, result in enumerate(

            results,

            start=1

        ):

            document = result["document"]

            score = result["score"]


            print(
                f"\nRank {rank}"
            )

            print(
                f"Score = {score:.3f}"
            )

            print(
                f"Document = {document['name']}"
            )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        print(
            "\n===== GENERATING ANSWER ====="
        )


        try:

            answer = generate_knowledge_answer(

                query,

                results

            )


            print(
                "\n===== FINAL ANSWER ====="
            )

            print(answer)


        except Exception as e:

            print(
                "\nERROR:"
            )

            print(e)