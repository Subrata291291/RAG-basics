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
# 4. LOAD PRODUCTS
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
Price: {product['price']}
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
# 8. KNOWLEDGE EMBEDDINGS
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

def parse_price_value(value):
    """
    Convert values such as:
        20k     -> 20000
        20.5k   -> 20500
        20000   -> 20000
    """
    value = value.lower().replace(",", "").strip()

    if value.endswith("k"):
        return int(float(value[:-1]) * 1000)

    return int(float(value))


def extract_price(text):
    """
    Extract a PRICE/BUDGET amount without confusing karat with price.

    Examples:
        "under 20k"          -> 20000
        "below ₹30,000"      -> 30000
        "above 25k"          -> 25000
        "22K gold under 30k" -> 30000
        "20K gold ring"      -> None
    """

    text = text.lower().replace(",", "")

    # --------------------------------------------------------
    # First: explicit price/budget phrases.
    # This prevents "22K gold under 30k" from returning 22000.
    # --------------------------------------------------------

    price_patterns = [
        r"(?:under|below|less than|upto|up to)\s*(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?k?)\b",
        r"(?:above|over|more than|greater than)\s*(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?k?)\b",
        r"(?:price|cost|budget)\s*(?:of|is|:)?\s*(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?k?)\b",
        r"(?:₹|rs\.?\s*)(\d+(?:\.\d+)?k?)\b",
    ]

    for pattern in price_patterns:
        match = re.search(pattern, text)
        if match:
            return parse_price_value(match.group(1))

    # --------------------------------------------------------
    # Standalone 4-7 digit amount.
    # Do NOT interpret 18/20/22/24 followed by K as price,
    # because those are normally gold karats.
    # --------------------------------------------------------

    match = re.search(r"\b(\d{4,7})\b", text)

    if match:
        return int(match.group(1))

    return None


# 10. KARAT EXTRACTION
# ============================================================

def extract_karat(text):

    text = text.lower()


    # --------------------------------------------------------
    # Only treat "20K" as karat when followed by "gold"
    # or when explicitly talking about karat.
    # --------------------------------------------------------

    match = re.search(

        r"\b(18|20|22|24)\s*k\s*(?:gold|karat|purity)?\b",

        text

    )


    if not match:

        return None


    # --------------------------------------------------------
    # Check whether this is actually price context.
    # --------------------------------------------------------

    number = match.group(1)

    matched_text = match.group(0)


    if "gold" in matched_text:

        return number + "K"


    if "karat" in matched_text:

        return number + "K"


    # Standalone "22K" can be interpreted as karat.
    # But "20k" after price words should be price.

    price_words = [

        "under",

        "below",

        "less",

        "price",

        "cost",

        "rupee",

        "rupees",

        "₹",

        "rs",

        "above",

        "over",

        "more"

    ]


    for word in price_words:

        if word in text:

            return None


    return number + "K"


# ============================================================
# 11. EXTRACT FILTERS
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

    filters["karat"] = extract_karat(
        query
    )


    # ========================================================
    # PRICE
    # ========================================================

    price = extract_price(
        query
    )


    # ========================================================
    # MAX PRICE
    # ========================================================

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
# 12. DISPLAY FILTERS
# ============================================================

def display_filters(filters):

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


# ============================================================
# NORMALIZE VALUE
# ============================================================

def normalize_value(value):

    if value is None:
        return ""

    return str(value).strip().lower()


# ============================================================
# NORMALIZE KARAT
# ============================================================

def normalize_karat(value):

    if value is None:
        return ""

    value = str(value).strip().upper()

    if value.endswith("K"):
        return value

    if value.isdigit():
        return value + "K"

    return value


# ============================================================
# METADATA FILTER
# ============================================================

def filter_products(products, filters):

    results = []

    for product in products:

        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        if filters["category"]:

            product_category = normalize_value(
                product.get("category")
            )

            requested_category = normalize_value(
                filters["category"]
            )

            if product_category != requested_category:
                continue

        # ----------------------------------------------------
        # METAL
        # ----------------------------------------------------

        if filters["metal"]:

            product_metal = normalize_value(
                product.get("metal")
            )

            requested_metal = normalize_value(
                filters["metal"]
            )

            if product_metal != requested_metal:
                continue

        # ----------------------------------------------------
        # KARAT
        # ----------------------------------------------------

        if filters["karat"]:

            product_karat = normalize_karat(
                product.get("karat")
            )

            requested_karat = normalize_karat(
                filters["karat"]
            )

            if product_karat != requested_karat:
                continue

        # ----------------------------------------------------
        # MAX PRICE
        # ----------------------------------------------------

        if filters["max_price"] is not None:

            product_price = int(product["price"])

            if product_price > filters["max_price"]:
                continue

        # ----------------------------------------------------
        # MIN PRICE
        # ----------------------------------------------------

        if filters["min_price"] is not None:

            product_price = int(product["price"])

            if product_price < filters["min_price"]:
                continue

        results.append(product)

    return results


# ============================================================
# 14. SEMANTIC PRODUCT SEARCH
# ============================================================

def semantic_search(query, filtered_products, top_k=3):

    if not filtered_products:
        return []

    # --------------------------------------------------------
    # Map product ID -> original embedding index.
    # Using IDs is safer than products.index(product).
    # --------------------------------------------------------

    product_index_by_id = {
        product["id"]: index
        for index, product in enumerate(products)
    }

    original_indexes = []

    for product in filtered_products:

        product_id = product["id"]

        if product_id in product_index_by_id:
            original_indexes.append(
                product_index_by_id[product_id]
            )

    if not original_indexes:
        return []

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(query)

    # --------------------------------------------------------
    # ONLY use embeddings belonging to metadata-filtered
    # products.
    # --------------------------------------------------------

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

            "product": filtered_products[index],

            "score": float(
                similarities[index]
            )

        })

    return results


# 15. CREATE PRODUCT CONTEXT
# ============================================================

def create_product_context(results):

    context = ""


    for result in results:

        product = result["product"]


        context += f"""

PRODUCT ID: {product['id']}

PRODUCT NAME: {product['name']}

CATEGORY: {product['category']}

METAL: {product['metal']}

KARAT: {product['karat']}

PRICE: ₹{product['price']:,}

DESCRIPTION:
{product['description']}

"""


    return context


# ============================================================
# 16. GENERATE PRODUCT ANSWER
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
the product data in the context.

IMPORTANT:

The product context is the source of truth.

STRICT RULES:

1. Never invent a product.

2. Never invent a price.

3. Never invent a karat.

4. Never invent a feature.

5. Never invent availability.

6. Never invent a product ID.

7. Never use information outside the context.

8. Do not make assumptions.

9. If the retrieved PRODUCT CONTEXT contains one or more matching products,
   NEVER say that the requested product does not exist.

10. Treat the retrieved PRODUCT CONTEXT as the authoritative result set.

11. If the customer asks to "show", "find", "list", or "give me" products,
    list the matching products from the context instead of claiming that
    the product does not exist.

12. Only recommend products present in the context.

11. Every price mentioned must exactly match
    a price in the context.

12. Every product name mentioned must exactly
    correspond to a product in the context.

13. Every product ID mentioned must exist
    in the context.

14. Keep the answer concise.

15. Do not mention RAG, embeddings,
    vector search, metadata filtering,
    validation or internal systems.

CUSTOMER QUESTION:

{query}

PRODUCT CONTEXT:

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
# 17. VALIDATION HELPERS
# ============================================================

def normalize_text(text):

    return text.lower().strip()


# ============================================================
# 18. VALIDATE PRODUCT NAMES
# ============================================================

def validate_product_names(answer, results):

    answer_lower = normalize_text(answer)

    allowed_names = {
        normalize_text(result["product"]["name"])
        for result in results
    }

    # Check every catalog product name that appears in answer.
    for product in products:

        name = normalize_text(product["name"])

        if name in answer_lower:

            if name not in allowed_names:

                return False, (
                    f"Product '{product['name']}' was mentioned "
                    f"but was not in retrieved context."
                )

    return True, "Product names valid."


# ============================================================
# 19. VALIDATE PRODUCT IDS
# ============================================================

def validate_product_ids(answer, results):

    answer_upper = answer.upper()

    allowed_ids = {
        result["product"]["id"].upper()
        for result in results
    }

    mentioned_ids = re.findall(
        r"\b[A-Z]\d{3}\b",
        answer_upper
    )

    for product_id in mentioned_ids:

        if product_id not in allowed_ids:

            return False, (
                f"Product ID '{product_id}' "
                f"is not in retrieved context."
            )

    return True, "Product IDs valid."


# ============================================================
# 20. VALIDATE PRODUCT PRICES
# ============================================================

def validate_prices(answer, results):
    """
    Validate product prices without treating a customer budget
    such as "under ₹20,000" as a product price.

    Strategy:
      1. Find explicit catalog product names in the answer.
      2. Look only near each product name.
      3. If a nearby price is present, it must equal that
         product's catalog price.
      4. Standalone budget numbers are ignored.

    This prevents false failures such as:
        "gold rings under ₹20,000"
    """

    answer_lower = answer.lower()

    for result in results:

        product = result["product"]

        product_name = product["name"].lower()

        expected_price = int(product["price"])

        position = answer_lower.find(product_name)

        if position == -1:
            continue

        # Look at a reasonably small window around the product.
        surrounding_text = answer_lower[
            max(0, position - 40):
            position + len(product_name) + 120
        ]

        price_strings = re.findall(
            r"(?:₹|rs\.?\s*)\s*[\d,]+",
            surrounding_text
        )

        for price_string in price_strings:

            number = re.sub(
                r"[^\d]",
                "",
                price_string
            )

            if not number:
                continue

            mentioned_price = int(number)

            # Ignore the customer's budget/filter.
            # Only reject an incorrect amount when it is close
            # enough to a specific retrieved product name.
            if mentioned_price != expected_price:

                # Common case:
                # "Gold rings under ₹20,000:
                #  Classic 22K Gold Ring - ₹18,000"
                #
                # The budget may be in the same window, so only
                # treat a price as a product price when the price
                # is after the product name.
                price_position = surrounding_text.find(
                    price_string
                )

                product_position = surrounding_text.find(
                    product_name
                )

                if price_position > product_position:

                    return False, (
                        f"Price ₹{mentioned_price:,} for "
                        f"'{product['name']}' does not match "
                        f"catalog price ₹{expected_price:,}."
                    )

    return True, "Product prices valid."


# ============================================================
# 21. VALIDATE KARATS
# ============================================================

def validate_karats(answer, results):

    answer_upper = answer.upper()

    allowed_karats = {
        result["product"]["karat"].upper()
        for result in results
    }

    # --------------------------------------------------------
    # Validate only explicit karat expressions.
    # --------------------------------------------------------

    mentioned_karats = re.findall(
        r"\b(18K|20K|22K|24K)\b",
        answer_upper
    )

    for karat in mentioned_karats:

        if karat not in allowed_karats:

            # A karat may appear as part of the user's request
            # echoed by the assistant. To remain strict, reject
            # only when it is presented as a product attribute.
            return False, (
                f"Karat '{karat}' "
                f"is not present in retrieved context."
            )

    return True, "Karats valid."


# ============================================================
# 22. COMPLETE ANSWER VALIDATION
# ============================================================

def validate_answer(answer, results):

    checks = [

        validate_product_names(
            answer,
            results
        ),

        validate_product_ids(
            answer,
            results
        ),

        validate_prices(
            answer,
            results
        ),

        validate_karats(
            answer,
            results
        )

    ]

    for valid, message in checks:

        if not valid:

            return False, message

    return True, "Answer passed validation."


# ============================================================
# 23. FALLBACK ANSWER
# ============================================================

def fallback_answer(results):

    if not results:

        return (
            "I couldn't find any matching products "
            "in the catalog."
        )

    lines = [
        "Here are the matching products "
        "available in the catalog:"
    ]

    for result in results:

        product = result["product"]

        lines.append(
            f"- {product['name']} "
            f"({product['karat']}) "
            f"- ₹{product['price']:,}"
        )

    return "\n".join(lines)


# 24. KNOWLEDGE SEARCH
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
                float(
                    similarities[index]
                )

        })


    return results


# ============================================================
# 25. KNOWLEDGE CONTEXT
# ============================================================

def create_knowledge_context(
    results
):

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
# 26. GENERATE KNOWLEDGE ANSWER
# ============================================================

def generate_knowledge_answer(

    query,

    results

):

    context = create_knowledge_context(
        results
    )


    prompt = f"""
You are a customer support assistant
for a jewellery website.

Use ONLY the information in the documents.

STRICT RULES:

- Do not invent policies.
- Do not invent return conditions.
- Do not invent refund times.
- Do not invent shipping times.
- Do not invent privacy practices.
- Do not use outside knowledge.
- If the answer is not contained in the documents,
  clearly say that the provided information
  does not contain the answer.

CUSTOMER QUESTION:

{query}

DOCUMENTS:

{context}

Answer clearly and briefly.
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
# 27. CONVERSATION MEMORY
# ============================================================

# Keep only a small amount of recent history.
# This is SHORT-TERM memory for the current terminal session.
conversation_history = []

# The most recent product/knowledge results are kept separately.
# This allows follow-up questions such as:
#   "Which one is best for a wedding?"
#   "What is the price of the first one?"
#   "Tell me more about the second one?"
last_product_results = []
last_knowledge_results = []

# CURRENT PRODUCT SEARCH STATE
# This stores the user's active product requirements across turns.
# Example:
#   "gold rings under 30k"
#   -> category=ring, metal=gold, max_price=30000
#
# Next:
#   "only 22K"
#   -> category=ring, metal=gold, karat=22K, max_price=30000
#
# Next:
#   "which is cheapest?"
#   -> searches the same filtered set.
current_product_filters = {
    "category": None,
    "metal": None,
    "karat": None,
    "max_price": None,
    "min_price": None
}


# ============================================================
# 28. FOLLOW-UP / REFINEMENT DETECTION
# ============================================================

def is_product_follow_up(query):

    text = query.lower().strip()

    phrases = [

        # References to previous results
        "which one",
        "which is best",
        "which is better",
        "the first one",
        "the second one",
        "the third one",
        "first one",
        "second one",
        "third one",
        "this one",
        "that one",
        "these",
        "those",
        "either",
        "both",
        "same one",
        "cheaper one",
        "cheapest one",
        "more expensive one",
        "most expensive one",

        # Details about a previous product
        "tell me more",
        "more details",
        "more detail",
        "details about it",
        "details about this",
        "details about that",
        "how much is it",
        "what is its price",
        "what's its price",
        "what is the price of it",
        "is it available",
        "does it come",
        "does it have",
        "why this",
        "why that",

        # Refinements
        "only ",
        "just ",
        "show only",
        "remove ",
        "exclude ",
        "instead",
        "make it",
        "change it",
        "change the",
        "also ",
        "and only"
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


def is_product_refinement(query):

    """
    True when the user is changing or narrowing the existing
    product search rather than starting a completely unrelated
    product search.

    Examples:
        "only 22K"
        "only gold"
        "under 20k"
        "show me silver"
        "only rings"
    """

    text = query.lower().strip()

    refinement_words = [
        "only",
        "just",
        "under",
        "below",
        "less than",
        "upto",
        "up to",
        "above",
        "over",
        "more than",
        "greater than",
        "change",
        "instead",
        "exclude",
        "remove"
    ]

    has_known_filter = any(
        token in text
        for token in [
            "ring",
            "rings",
            "necklace",
            "necklaces",
            "bracelet",
            "bracelets",
            "earring",
            "earrings",
            "gold",
            "silver",
            "18k",
            "20k",
            "22k",
            "24k",
            "karat"
        ]
    )

    return (
        any(word in text for word in refinement_words)
        and has_known_filter
    )


def merge_product_filters(
    old_filters,
    new_filters
):
    """
    Merge only explicitly detected new requirements.

    This is the key state-management operation.

    Example:
        old = gold + ring + max_price 30000
        new = karat 22K

        result =
        gold + ring + 22K + max_price 30000
    """

    merged = old_filters.copy()

    for key in [
        "category",
        "metal",
        "karat",
        "max_price",
        "min_price"
    ]:

        if new_filters.get(key) is not None:
            merged[key] = new_filters[key]

    return merged


def clear_conflicting_price_filters(filters, query):

    """
    If the customer explicitly changes the price direction,
    remove the old opposite constraint.

    Example:
        old: under 30k
        new: above 20k

        result: above 20k
        (unless the user explicitly gives both constraints
        in the same sentence)
    """

    text = query.lower()

    has_max = any(
        phrase in text
        for phrase in [
            "under",
            "below",
            "less than",
            "upto",
            "up to"
        ]
    )

    has_min = any(
        phrase in text
        for phrase in [
            "above",
            "over",
            "more than",
            "greater than"
        ]
    )

    updated = filters.copy()

    if has_max and not has_min:
        updated["min_price"] = None

    if has_min and not has_max:
        updated["max_price"] = None

    return updated


# 29. BUILD CONVERSATION CONTEXT
# ============================================================

def build_conversation_context():

    if not conversation_history:
        return ""

    recent = conversation_history[-6:]

    lines = []

    for turn in recent:

        lines.append(
            f"Customer: {turn['user']}"
        )

        lines.append(
            f"Assistant: {turn['assistant']}"
        )

    return "\n".join(lines)


# ============================================================
# 30. REWRITE FOLLOW-UP QUESTION
# ============================================================

def rewrite_follow_up_question(query):

    """
    Convert a follow-up into a standalone query.

    Example:

        Previous:
        "Show me gold rings under ₹30,000."

        Current:
        "Which one is best for a wedding?"

        Rewritten:
        "Which of the previously shown gold rings under ₹30,000
         is best for a wedding?"

    IMPORTANT:
    This function is ONLY for understanding the conversation.
    It does not answer the customer.
    """

    if not conversation_history:
        return query

    context = build_conversation_context()

    prompt = f"""
Rewrite the customer's latest question into a standalone question
using ONLY facts explicitly present in the conversation.

Do not answer the question.

Do not invent products, prices, karats, features, policies,
availability, or other facts.

If the latest question is already standalone, return it unchanged.

If it refers to "it", "this", "that", "these", "those",
"the first one", "the second one", "which one", etc.,
resolve the reference using the conversation.

Conversation:

{context}

Latest customer question:

{query}

Return ONLY the rewritten standalone question.
"""

    try:

        response = client.chat.completions.create(

            model="openrouter/free",

            messages=[

                {
                    "role": "system",
                    "content":
                        "You rewrite follow-up questions "
                        "without adding facts."
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ]

        )

        rewritten = (
            response.choices[0]
            .message.content
            .strip()
        )

        # Basic safety:
        # never allow an empty rewrite.
        if rewritten:
            return rewritten

    except Exception:

        pass

    return query


# ============================================================
# 31. CREATE PRODUCT CONTEXT WITH RANK NUMBERS
# ============================================================

def create_numbered_product_context(results):

    context = ""

    for rank, result in enumerate(results, start=1):

        product = result["product"]

        context += f"""
PRODUCT {rank}

PRODUCT ID: {product['id']}
PRODUCT NAME: {product['name']}
CATEGORY: {product['category']}
METAL: {product['metal']}
KARAT: {product['karat']}
PRICE: ₹{product['price']:,}
DESCRIPTION:
{product['description']}

"""

    return context


# ============================================================
# 32. GENERATE ANSWER FROM PREVIOUS PRODUCTS
# ============================================================

def generate_follow_up_product_answer(
    query,
    previous_results
):

    context = create_numbered_product_context(
        previous_results
    )

    prompt = f"""
You are a jewellery shopping assistant.

The customer is asking a follow-up question about products
that were already shown.

Use ONLY the products in the context below.

STRICT RULES:

1. Never invent a product.
2. Never invent a price.
3. Never invent a karat.
4. Never invent a feature.
5. Never invent availability.
6. Never invent a product ID.
7. Do not use outside knowledge.
8. If the requested information is not present,
   say that it is not available in the provided product data.
9. If the customer asks "which one is best", choose based ONLY
   on the descriptions and attributes in the context.
10. If the customer asks for "the first", "second", or "third"
    product, use the numbered products exactly.
11. Every price mentioned must exactly match the context.
12. Keep the answer concise.

CUSTOMER FOLLOW-UP:

{query}

PREVIOUS PRODUCTS:

{context}

Answer the customer.
"""

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
# 33. GENERATE KNOWLEDGE FOLLOW-UP ANSWER
# ============================================================

def generate_follow_up_knowledge_answer(
    query,
    previous_results
):

    context = create_knowledge_context(
        previous_results
    )

    prompt = f"""
You are a customer support assistant for a jewellery website.

The customer is asking a follow-up question about the previously
retrieved policy/knowledge documents.

Use ONLY the document context below.

STRICT RULES:

- Do not invent policies.
- Do not invent return conditions.
- Do not invent refund times.
- Do not invent shipping times.
- Do not invent privacy practices.
- Do not use outside knowledge.
- If the answer is not contained in the documents,
  clearly say that the provided documents do not contain
  the answer.

CUSTOMER QUESTION:

{query}

DOCUMENT CONTEXT:

{context}

Answer clearly and briefly.
"""

    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[

            {
                "role": "system",
                "content":
                    "You are a strictly grounded customer "
                    "support assistant."
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )

    return response.choices[0].message.content


# ============================================================
# 34. ADD TO MEMORY
# ============================================================

def remember_turn(user_query, assistant_answer):

    conversation_history.append({

        "user": user_query,

        "assistant": assistant_answer

    })

    # Keep memory small.
    if len(conversation_history) > 6:

        del conversation_history[:-6]


# ============================================================
# 35. STATE-AWARE PRODUCT HELPERS
# ============================================================

def get_product_by_position(results, position):

    if not results:
        return None

    index = position - 1

    if 0 <= index < len(results):
        return results[index]["product"]

    return None


def answer_relative_product_question(
    query,
    results
):
    """
    Handle deterministic questions about:
      - first / second / third product
      - cheapest
      - most expensive

    This avoids asking the LLM to guess which product a reference
    means.
    """

    text = query.lower()

    # --------------------------------------------------------
    # Position references
    # --------------------------------------------------------

    position = None

    if "first" in text:
        position = 1

    elif "second" in text:
        position = 2

    elif "third" in text:
        position = 3

    if position is not None:

        product = get_product_by_position(
            results,
            position
        )

        if product is not None:

            return product

    # --------------------------------------------------------
    # Cheapest
    # --------------------------------------------------------

    if (
        "cheapest" in text
        or "lowest price" in text
        or "least expensive" in text
    ):

        if results:

            return min(
                (
                    result["product"]
                    for result in results
                ),
                key=lambda p: int(p["price"])
            )

    # --------------------------------------------------------
    # Most expensive
    # --------------------------------------------------------

    if (
        "most expensive" in text
        or "highest price" in text
        or "costliest" in text
    ):

        if results:

            return max(
                (
                    result["product"]
                    for result in results
                ),
                key=lambda p: int(p["price"])
            )

    return None


def create_single_product_context(product):

    return f"""
PRODUCT ID: {product['id']}
PRODUCT NAME: {product['name']}
CATEGORY: {product['category']}
METAL: {product['metal']}
KARAT: {product['karat']}
PRICE: ₹{product['price']:,}
DESCRIPTION:
{product['description']}
"""


def generate_single_product_answer(
    query,
    product
):

    context = create_single_product_context(
        product
    )

    prompt = f"""
You are a jewellery shopping assistant.

Answer the customer's question using ONLY this product data.

PRODUCT DATA:
{context}

STRICT RULES:
- Never invent information.
- Never invent a price.
- Never invent a feature.
- Never invent availability.
- Never invent a product ID.
- If the requested information is not in the product data,
  say that it is not available.
- Keep the answer concise.

CUSTOMER QUESTION:
{query}

Answer the customer.
"""

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


def reset_product_state():

    global current_product_filters
    global last_product_results

    current_product_filters = {
        "category": None,
        "metal": None,
        "karat": None,
        "max_price": None,
        "min_price": None
    }

    last_product_results = []


def update_product_state(
    query,
    extracted_filters
):

    global current_product_filters

    # If there is no active product state, simply start one.
    has_existing_state = any(
        value is not None
        for value in current_product_filters.values()
    )

    if not has_existing_state:

        current_product_filters = extracted_filters.copy()

        return current_product_filters

    # Merge new explicit filters into the old state.
    merged = merge_product_filters(
        current_product_filters,
        extracted_filters
    )

    # Handle changes in price direction.
    merged = clear_conflicting_price_filters(
        merged,
        query
    )

    current_product_filters = merged

    return current_product_filters


def display_current_state(filters):

    print(
        "\n===== CURRENT CONVERSATION STATE ====="
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


# ============================================================
# 36. MAIN CHATBOT
# ============================================================


# ============================================================

print()
print("=" * 60)
print("        PROJECT 26 - STATEFUL CONVERSATIONAL JEWELLERY RAG")
print("=" * 60)

print()
print("Products + Policies + Semantic Search + Validation")
print("Short-term conversation memory + persistent product state enabled.")

print()
print("Examples:")

print(
    "  Show me gold rings under ₹30,000"
)

print(
    "  Which one is best for a wedding?"
)

print(
    "  What is the price of the first one?"
)

print()
print("Type 'exit' to quit.")


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
    # EMPTY
    # ========================================================

    if not query:

        print(
            "Please enter a question."
        )

        continue


    # ========================================================
    # KNOWLEDGE / POLICY QUERY DETECTION
    # ========================================================

    text_lower = query.lower()

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

    is_knowledge_query = any(
        word in text_lower
        for word in policy_words
    )


    # ========================================================
    # KNOWLEDGE QUERY
    # ========================================================

    if is_knowledge_query:

        print(
            "\nQuery type: knowledge"
        )

        results = search_knowledge(
            query,
            top_k=2
        )

        last_knowledge_results = results

        # A knowledge question does not replace the product
        # state. The customer can come back to it later.
        print(
            "\n===== KNOWLEDGE SEARCH ====="
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRank {rank}"
            )

            print(
                f"Score = "
                f"{result['score']:.3f}"
            )

            print(
                "Document = "
                f"{result['document']['name']}"
            )

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

            remember_turn(
                query,
                answer
            )

        except Exception as e:

            print(
                "\nERROR:"
            )

            print(e)

        continue


    # ========================================================
    # PRODUCT QUERY / FOLLOW-UP
    # ========================================================

    follow_up = (
        is_product_follow_up(query)
        or is_product_refinement(query)
    )

    # --------------------------------------------------------
    # CASE 1: Relative question about currently displayed
    # products.
    #
    # Examples:
    #   "what is the price of the first one?"
    #   "tell me more about the second one"
    #   "which is cheapest?"
    # --------------------------------------------------------

    relative_product = None

    if follow_up and last_product_results:

        relative_product = answer_relative_product_question(
            query,
            last_product_results
        )

    if relative_product is not None:

        print(
            "\nQuery type: product follow-up"
        )

        print(
            "\n===== REFERENCED PRODUCT ====="
        )

        print(
            f"ID: {relative_product['id']}"
        )

        print(
            f"Name: {relative_product['name']}"
        )

        print(
            f"Price: ₹{relative_product['price']:,}"
        )

        print(
            "\n===== GENERATING ANSWER ====="
        )

        try:

            answer = generate_single_product_answer(
                query,
                relative_product
            )

            print(
                "\n===== GENERATED ANSWER ====="
            )

            print(answer)

            # Validate against a one-product context.
            one_result = [{
                "product": relative_product,
                "score": 1.0
            }]

            is_valid, validation_message = validate_answer(
                answer,
                one_result
            )

            print(
                "\n===== ANSWER VALIDATION ====="
            )

            if is_valid:

                print("VALIDATION: PASSED")
                print(validation_message)

                print(
                    "\n===== FINAL ANSWER ====="
                )

                print(answer)

                remember_turn(
                    query,
                    answer
                )

            else:

                print("VALIDATION: FAILED")
                print(validation_message)

                safe_answer = fallback_answer(
                    one_result
                )

                print(
                    "\n===== FINAL ANSWER ====="
                )

                print(safe_answer)

                remember_turn(
                    query,
                    safe_answer
                )

        except Exception as e:

            print(
                "\nERROR:"
            )

            print(e)

        continue


    # ========================================================
    # CASE 2: Product refinement / new product search
    # ========================================================

    print(
        "\nQuery type: product"
    )

    # --------------------------------------------------------
    # Extract only the filters explicitly present in THIS turn.
    # --------------------------------------------------------

    new_filters = extract_filters(query)

    # --------------------------------------------------------
    # If this is a new unrelated product query, start fresh.
    # If it is a refinement, merge with previous state.
    # --------------------------------------------------------

    has_state = any(
        value is not None
        for value in current_product_filters.values()
    )

    if (
        has_state
        and (
            is_product_refinement(query)
            or follow_up
        )
    ):

        filters = update_product_state(
            query,
            new_filters
        )

    else:

        # New standalone product search.
        current_product_filters = new_filters.copy()
        filters = current_product_filters.copy()


    display_filters(filters)
    display_current_state(filters)


    # ========================================================
    # METADATA FILTER
    # ========================================================

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
            f"{product['metal']} | "
            f"{product['karat']} | "
            f"₹{product['price']:,}"
        )


    # ========================================================
    # NO MATCH
    # ========================================================

    if not filtered_products:

        print(
            "\n===== FINAL ANSWER ====="
        )

        answer = (
            "I couldn't find any products "
            "matching your current requirements."
        )

        print(answer)

        remember_turn(
            query,
            answer
        )

        # Keep the state so the customer can correct/refine it.
        last_product_results = []

        continue


    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    # For a refinement, search using the full conversation state
    # plus the current request. This helps "only 22K" while still
    # preserving the earlier category/metal/price constraints.
    search_query = query

    if has_state and (
        is_product_refinement(query)
        or follow_up
    ):

        state_parts = []

        if filters["category"]:
            state_parts.append(
                f"category {filters['category']}"
            )

        if filters["metal"]:
            state_parts.append(
                f"metal {filters['metal']}"
            )

        if filters["karat"]:
            state_parts.append(
                f"karat {filters['karat']}"
            )

        if filters["max_price"] is not None:
            state_parts.append(
                f"maximum price {filters['max_price']}"
            )

        if filters["min_price"] is not None:
            state_parts.append(
                f"minimum price {filters['min_price']}"
            )

        if state_parts:

            search_query = (
                query
                + "\nCurrent customer requirements: "
                + ", ".join(state_parts)
            )


    results = semantic_search(
        search_query,
        filtered_products,
        top_k=3
    )


    # ========================================================
    # FINAL RETRIEVAL SAFETY CHECK
    # ========================================================

    allowed_product_ids = {
        product["id"]
        for product in filtered_products
    }

    results = [
        result
        for result in results
        if result["product"]["id"]
        in allowed_product_ids
    ]


    # Store latest results.
    last_product_results = results

    # Product question means the next knowledge follow-up should
    # not accidentally use an old policy result.
    last_knowledge_results = []


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(
        "\n===== FINAL RETRIEVED PRODUCTS ====="
    )


    for rank, result in enumerate(
        results,
        start=1
    ):

        product = result["product"]

        print(
            f"\nRank {rank}"
        )

        print(
            f"Semantic Score = "
            f"{result['score']:.3f}"
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


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    print(
        "\n===== GENERATING ANSWER ====="
    )


    try:

        answer = generate_product_answer(
            query,
            results
        )

        # ====================================================
        # MODEL OUTPUT SANITY CHECK
        # ====================================================
        # If retrieval found products, the model must not claim
        # that no such product exists. This protects against
        # occasional free-model hallucinations.
        # ====================================================

        no_product_claims = [
            "no product",
            "there is no product",
            "there's no product",
            "couldn't find any products",
            "could not find any products",
            "no products matching"
        ]

        answer_lower = answer.lower()

        if results and any(
            phrase in answer_lower
            for phrase in no_product_claims
        ):

            answer = fallback_answer(results)


        print(
            "\n===== GENERATED ANSWER ====="
        )

        print(answer)


        # ====================================================
        # VALIDATION
        # ====================================================

        print(
            "\n===== ANSWER VALIDATION ====="
        )


        is_valid, validation_message = validate_answer(
            answer,
            results
        )


        if is_valid:

            print(
                "VALIDATION: PASSED"
            )

            print(
                validation_message
            )

            print(
                "\n===== FINAL ANSWER ====="
            )

            print(answer)

            remember_turn(
                query,
                answer
            )


        else:

            print(
                "VALIDATION: FAILED"
            )

            print(
                validation_message
            )

            print(
                "\nUsing safe fallback answer..."
            )

            safe_answer = fallback_answer(
                results
            )

            print(
                "\n===== FINAL ANSWER ====="
            )

            print(safe_answer)

            remember_turn(
                query,
                safe_answer
            )


    except Exception as e:

        print(
            "\nERROR:"
        )

        print(e)
