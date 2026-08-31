import os
import re
from copy import deepcopy

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT 27
# PRODUCT RESULT SET + PAGINATION
# ============================================================
#
# Goal:
#   - Keep ALL products that match metadata filters.
#   - Semantically rank the matching products.
#   - Keep the complete result set in conversation state.
#   - Show only a page of products at a time.
#   - Support follow-ups such as:
#       "show more"
#       "which one is cheapest?"
#       "only 22K"
#       "sort by price"
#       "tell me more about the second one"
#
# Important:
#   The catalog/filter layer is the source of truth.
#   The LLM is used for natural-language responses, not
#   for deciding whether a product exists.
# ============================================================


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("ERROR: OPENROUTER_API_KEY not found.")
    print("Please check your .env file.")
    raise SystemExit(1)


# ============================================================
# 2. OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ============================================================
# 3. EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


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
        "description": "An elegant 22K gold ring with a beautiful traditional design, suitable for weddings and special occasions."
    },
    {
        "id": "R003",
        "name": "Traditional 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 27000,
        "description": "A traditional 22K gold ring with an ornate design suitable for weddings."
    },
    {
        "id": "R004",
        "name": "Bridal 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 29000,
        "description": "A bridal 22K gold ring designed for weddings and special occasions."
    },
    {
        "id": "R005",
        "name": "Modern 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "22K",
        "price": 24000,
        "description": "A modern 22K gold ring with a minimal contemporary design suitable for parties and special occasions."
    },
    {
        "id": "R006",
        "name": "Diamond Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": "18K",
        "price": 19500,
        "description": "An 18K gold diamond ring with a modern design, ideal for special occasions and weddings."
    },
    {
        "id": "S001",
        "name": "Silver Ring",
        "category": "ring",
        "metal": "silver",
        "karat": "0K",
        "price": 8000,
        "description": "A simple silver ring suitable for everyday wear."
    },
    {
        "id": "N001",
        "name": "22K Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": "22K",
        "price": 35000,
        "description": "A traditional 22K gold necklace suitable for festive occasions."
    },
    {
        "id": "N002",
        "name": "Diamond Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": "18K",
        "price": 45000,
        "description": "An elegant 18K gold diamond necklace for special occasions."
    },
    {
        "id": "B001",
        "name": "22K Gold Bracelet",
        "category": "bracelet",
        "metal": "gold",
        "karat": "22K",
        "price": 22000,
        "description": "A 22K gold bracelet with a traditional design suitable for festive occasions."
    },
    {
        "id": "E001",
        "name": "Gold Earrings",
        "category": "earrings",
        "metal": "gold",
        "karat": "22K",
        "price": 15000,
        "description": "22K gold earrings with an elegant design suitable for weddings and festive occasions."
    }
]


# ============================================================
# 5. CREATE SEARCH TEXT + EMBEDDINGS
# ============================================================

def product_search_text(product):
    return (
        f"{product['name']}. "
        f"Category: {product['category']}. "
        f"Metal: {product['metal']}. "
        f"Karat: {product['karat']}. "
        f"Price: ₹{product['price']}. "
        f"{product['description']}"
    )


print("Creating product embeddings...")

product_texts = [
    product_search_text(product)
    for product in products
]

product_embeddings = embedding_model.encode(
    product_texts,
    normalize_embeddings=True
)

print("Product embeddings created.")


# ============================================================
# 6. CONVERSATION STATE
# ============================================================

state = {
    "filters": {
        "category": None,
        "metal": None,
        "karat": None,
        "min_price": None,
        "max_price": None
    },

    # Complete matching set after metadata filtering + ranking.
    "result_set": [],

    # Current page number.
    "page": 1,

    # Number of products displayed per page.
    "page_size": 5,

    # Last products displayed to the user.
    "last_displayed_ids": [],

    # Last referenced product.
    "last_referenced_product_id": None
}


# ============================================================
# 7. NORMALIZATION
# ============================================================

def normalize_text(text):
    text = text.lower().strip()

    replacements = {
        "jewellery": "jewelry",
        "rings": "ring",
        "necklaces": "necklace",
        "bracelets": "bracelet",
        "earrings": "earring",
        "kgs": "kg",
        "under": "under",
    }

    for old, new in replacements.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)

    return text


# ============================================================
# 8. PRICE EXTRACTION
# ============================================================

def extract_price_value(text):
    """
    Supports:
        20k
        30K
        30000
        ₹30,000
        30,000
    """

    match = re.search(
        r"(?:₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(k|thousand)?",
        text.lower()
    )

    if not match:
        return None

    number = float(match.group(1).replace(",", ""))
    suffix = match.group(2)

    if suffix == "k":
        number *= 1000
    elif suffix == "thousand":
        number *= 1000

    return int(number)


def extract_price_filters(query):
    q = query.lower()

    min_price = None
    max_price = None

    # "under 30000", "below 30000", "less than 30000"
    match = re.search(
        r"(?:under|below|less than|up to|max(?:imum)?(?: price)?(?: of)?)\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand)?",
        q
    )

    if match:
        number = float(match.group(1).replace(",", ""))
        suffix = match.group(2)

        if suffix == "k":
            number *= 1000
        elif suffix == "thousand":
            number *= 1000

        max_price = int(number)

    # "above 20000", "over 20000", "more than 20000"
    match = re.search(
        r"(?:above|over|more than|greater than|from)\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand)?",
        q
    )

    if match:
        number = float(match.group(1).replace(",", ""))
        suffix = match.group(2)

        if suffix == "k":
            number *= 1000
        elif suffix == "thousand":
            number *= 1000

        min_price = int(number)

    # "between 20000 and 30000"
    match = re.search(
        r"between\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand)?\s*and\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand)?",
        q
    )

    if match:
        first = float(match.group(1).replace(",", ""))
        second = float(match.group(3).replace(",", ""))

        suffix1 = match.group(2)
        suffix2 = match.group(4)

        if suffix1 == "k":
            first *= 1000
        elif suffix1 == "thousand":
            first *= 1000

        if suffix2 == "k":
            second *= 1000
        elif suffix2 == "thousand":
            second *= 1000

        min_price = int(min(first, second))
        max_price = int(max(first, second))

    return min_price, max_price


# ============================================================
# 9. FILTER EXTRACTION
# ============================================================

def extract_filters(query, previous_filters):
    """
    Update only the filters explicitly mentioned in the new query.

    This is what allows:
        "show me gold rings under 30000"
        -> "only 22K"
        -> "under 25000"

    to maintain conversation state.
    """

    q = normalize_text(query)

    filters = deepcopy(previous_filters)

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    categories = {
        "ring": ["ring"],
        "necklace": ["necklace"],
        "bracelet": ["bracelet"],
        "earrings": ["earring", "earrings"]
    }

    for category, words in categories.items():
        if any(re.search(rf"\b{re.escape(word)}\b", q) for word in words):
            filters["category"] = category
            break

    # --------------------------------------------------------
    # METAL
    # --------------------------------------------------------

    if re.search(r"\bgold\b", q):
        filters["metal"] = "gold"
    elif re.search(r"\bsilver\b", q):
        filters["metal"] = "silver"

    # --------------------------------------------------------
    # KARAT
    # --------------------------------------------------------

    karat_match = re.search(
        r"\b(9|14|18|22|24)\s*k(?:arat)?\b",
        q
    )

    if karat_match:
        filters["karat"] = f"{karat_match.group(1)}K"

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    min_price, max_price = extract_price_filters(query)

    if min_price is not None:
        filters["min_price"] = min_price

    if max_price is not None:
        filters["max_price"] = max_price

    return filters


# ============================================================
# 10. METADATA FILTERING
# ============================================================

def metadata_filter(filters):
    matches = []

    for product in products:

        if (
            filters["category"] is not None
            and product["category"] != filters["category"]
        ):
            continue

        if (
            filters["metal"] is not None
            and product["metal"] != filters["metal"]
        ):
            continue

        if (
            filters["karat"] is not None
            and product["karat"] != filters["karat"]
        ):
            continue

        if (
            filters["min_price"] is not None
            and product["price"] < filters["min_price"]
        ):
            continue

        if (
            filters["max_price"] is not None
            and product["price"] > filters["max_price"]
        ):
            continue

        matches.append(product)

    return matches


# ============================================================
# 11. SEMANTIC RANKING
# ============================================================

def semantic_rank(query, candidates):
    if not candidates:
        return []

    candidate_texts = [
        product_search_text(product)
        for product in candidates
    ]

    candidate_embeddings = embedding_model.encode(
        candidate_texts,
        normalize_embeddings=True
    )

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        candidate_embeddings
    )[0]

    ranked = []

    for product, score in zip(candidates, scores):
        item = deepcopy(product)
        item["semantic_score"] = float(score)
        ranked.append(item)

    ranked.sort(
        key=lambda item: item["semantic_score"],
        reverse=True
    )

    return ranked


# ============================================================
# 12. SORTING
# ============================================================

def sort_results(results, query):
    q = query.lower()

    if "cheapest" in q or "lowest price" in q or "low to high" in q:
        return sorted(results, key=lambda x: x["price"])

    if "most expensive" in q or "highest price" in q or "high to low" in q:
        return sorted(results, key=lambda x: x["price"], reverse=True)

    return results


# ============================================================
# 13. PRODUCT RESULT SET
# ============================================================

def create_result_set(query, filters):
    """
    IMPORTANT:
    We do NOT cut the result set to top 3 here.

    Every metadata-matching product is preserved.
    Semantic ranking only determines ordering.
    """

    matching_products = metadata_filter(filters)

    ranked_products = semantic_rank(
        query,
        matching_products
    )

    ranked_products = sort_results(
        ranked_products,
        query
    )

    return ranked_products


# ============================================================
# 14. PAGINATION
# ============================================================

def get_current_page():
    result_set = state["result_set"]

    page_size = state["page_size"]

    start = (state["page"] - 1) * page_size
    end = start + page_size

    return result_set[start:end]


def has_more_pages():
    page_size = state["page_size"]

    next_start = state["page"] * page_size

    return next_start < len(state["result_set"])


def show_next_page():
    if not has_more_pages():
        return []

    state["page"] += 1

    return get_current_page()


# ============================================================
# 15. DISPLAY PRODUCTS
# ============================================================

def display_products(products_to_show, title="PRODUCTS"):
    print(f"\n===== {title} =====")

    if not products_to_show:
        print("No products to display.")
        return

    state["last_displayed_ids"] = [
        product["id"]
        for product in products_to_show
    ]

    total = len(state["result_set"])
    start = (state["page"] - 1) * state["page_size"] + 1
    end = min(
        start + len(products_to_show) - 1,
        total
    )

    print(
        f"Showing {start}-{end} of {total} matching products."
    )

    for index, product in enumerate(products_to_show, start=1):

        print(f"\n{index}. {product['name']}")
        print(f"   ID: {product['id']}")
        print(f"   Metal: {product['metal']}")
        print(f"   Karat: {product['karat']}")
        print(f"   Price: ₹{product['price']:,}")
        print(
            f"   Semantic Score: "
            f"{product.get('semantic_score', 0):.3f}"
        )
        print(f"   {product['description']}")

    if has_more_pages():
        print("\nType 'show more' to see more products.")


# ============================================================
# 16. PRODUCT LOOKUP
# ============================================================

def find_product_in_current_results(identifier_or_number):
    """
    Find a product from the CURRENT result set.

    Examples:
        "first one"
        "second one"
        "R002"
    """

    value = identifier_or_number.lower().strip()

    ordinal_map = {
        "first": 0,
        "1st": 0,
        "second": 1,
        "2nd": 1,
        "third": 2,
        "3rd": 2,
        "fourth": 3,
        "4th": 3,
        "fifth": 4,
        "5th": 4
    }

    if value in ordinal_map:
        index = ordinal_map[value]

        page = get_current_page()

        if index < len(page):
            return page[index]

        return None

    for product in state["result_set"]:
        if product["id"].lower() == value:
            return product

    return None


# ============================================================
# 17. DETECT FOLLOW-UP COMMANDS
# ============================================================

def is_show_more(query):
    q = query.lower().strip()

    phrases = [
        "show more",
        "more products",
        "next page",
        "show next",
        "see more",
        "more"
    ]

    return any(
        phrase == q or phrase in q
        for phrase in phrases
    )


def is_cheapest_question(query):
    q = query.lower()

    return (
        "cheapest" in q
        or "lowest price" in q
        or "least expensive" in q
    )


def is_most_expensive_question(query):
    q = query.lower()

    return (
        "most expensive" in q
        or "highest price" in q
        or "costliest" in q
    )


# ============================================================
# 18. SAFE ANSWERS WITHOUT LLM
# ============================================================

def answer_cheapest():
    if not state["result_set"]:
        return "There are no matching products."

    product = min(
        state["result_set"],
        key=lambda item: item["price"]
    )

    state["last_referenced_product_id"] = product["id"]

    return (
        f"The cheapest matching product is "
        f"{product['name']} ({product['id']}) at "
        f"₹{product['price']:,}.\n"
        f"{product['description']}"
    )


def answer_most_expensive():
    if not state["result_set"]:
        return "There are no matching products."

    product = max(
        state["result_set"],
        key=lambda item: item["price"]
    )

    state["last_referenced_product_id"] = product["id"]

    return (
        f"The most expensive matching product is "
        f"{product['name']} ({product['id']}) at "
        f"₹{product['price']:,}.\n"
        f"{product['description']}"
    )


# ============================================================
# 19. LLM GENERATION
# ============================================================

def deterministic_product_answer(query, displayed_products):
    """
    For product-listing/search queries, do NOT depend on a free
    LLM to decide what to say.

    The catalog is the source of truth, so we format the answer
    directly from the retrieved products.

    This prevents free OpenRouter models from returning unrelated
    outputs such as:
        "User Safety: safe"
    """

    if not displayed_products:
        return "I couldn't find any products matching your requirements."

    lines = []

    if len(displayed_products) == 1:
        lines.append("Here is the matching product from our catalog:")
    else:
        lines.append("Here are the matching products from our catalog:")

    for product in displayed_products:
        lines.append(
            f"- {product['name']} ({product['id']}) — "
            f"₹{product['price']:,}"
        )
        lines.append(
            f"  {product['description']}"
        )

    if has_more_pages():
        lines.append("")
        lines.append(
            "Type 'show more' to see the next products."
        )

    return "\n".join(lines)


def generate_answer(query, displayed_products):
    """
    Use deterministic catalog output for product searches.

    The free OpenRouter model is intentionally NOT used for the
    basic product-listing response. This makes the result reliable.

    The LLM can be added later for conversational explanations,
    but it should never be responsible for deciding the catalog
    result.
    """

    return deterministic_product_answer(
        query,
        displayed_products
    )


# ============================================================
# 20. SIMPLE ANSWER VALIDATION
# ============================================================

# ============================================================

def validate_answer(answer, displayed_products):
    """
    Stronger validation.

    Reject:
      - empty answers
      - unrelated safety/classification responses
      - prices not present in the supplied catalog context
      - product IDs not present in the supplied context

    Accept:
      - deterministic catalog answers
      - factual answers containing only supplied product data
    """

    if not answer or not answer.strip():
        return False

    lower_answer = answer.lower().strip()

    # --------------------------------------------------------
    # Reject obvious non-product/model-classification output
    # --------------------------------------------------------

    forbidden_phrases = [
        "user safety:",
        "safety: safe",
        "safety: unsafe",
        "classification:",
        "content policy",
        "i cannot help",
        "i can't help",
        "as an ai language model"
    ]

    if any(
        phrase in lower_answer
        for phrase in forbidden_phrases
    ):
        return False

    # --------------------------------------------------------
    # Build allowed values from current displayed products
    # --------------------------------------------------------

    valid_prices = {
        f"₹{p['price']:,}"
        for p in displayed_products
    }

    valid_ids = {
        p["id"].lower()
        for p in displayed_products
    }

    valid_names = {
        p["name"].lower()
        for p in displayed_products
    }

    # --------------------------------------------------------
    # Check prices
    # --------------------------------------------------------

    found_prices = re.findall(
        r"₹\s*[\d,]+",
        answer
    )

    for price in found_prices:
        normalized = price.replace("₹ ", "₹")

        if normalized not in valid_prices:
            return False

    # --------------------------------------------------------
    # Check product IDs
    # --------------------------------------------------------

    found_ids = re.findall(
        r"\b[A-Z]\d{3}\b",
        answer,
        flags=re.IGNORECASE
    )

    for product_id in found_ids:
        if product_id.lower() not in valid_ids:
            return False

    # --------------------------------------------------------
    # For a product-listing answer, require at least one
    # displayed product name or ID.
    # --------------------------------------------------------

    mentions_product = any(
        product_id in lower_answer
        for product_id in valid_ids
    ) or any(
        name in lower_answer
        for name in valid_names
    )

    if not mentions_product:
        return False

    return True


# ============================================================
# 21. SAFE FALLBACK
# ============================================================

def fallback_answer(displayed_products):
    if not displayed_products:
        return "I couldn't find any products matching your requirements."

    lines = [
        "Here are the matching products from our catalog:"
    ]

    for product in displayed_products:
        lines.append(
            f"- {product['name']} ({product['id']}) — "
            f"₹{product['price']:,}"
        )

    return "\n".join(lines)


# ============================================================
# 22. SHOW CURRENT STATE
# ============================================================

def display_state():
    filters = state["filters"]

    print("\n===== CURRENT SEARCH STATE =====")

    print(f"category  -> {filters['category']}")
    print(f"metal     -> {filters['metal']}")
    print(f"karat     -> {filters['karat']}")
    print(f"min_price -> {filters['min_price']}")
    print(f"max_price -> {filters['max_price']}")

    print(
        f"Total matching products -> "
        f"{len(state['result_set'])}"
    )

    print(
        f"Current page -> "
        f"{state['page']}"
    )


# ============================================================
# 23. NEW PRODUCT SEARCH
# ============================================================

def perform_new_product_search(query):
    # --------------------------------------------------------
    # EXPLICIT NEW SEARCH DETECTION
    # --------------------------------------------------------
    #
    # If the user explicitly names a new category or metal,
    # treat it as a new product search rather than carrying
    # conflicting old values.
    #
    # Example:
    #   "show me gold rings"
    #   "show me silver rings"
    #
    # The second query must not inherit metal=gold.
    # --------------------------------------------------------

    query_lower = normalize_text(query)

    explicit_category = any(
        word in query_lower.split()
        for word in ["ring", "necklace", "bracelet", "earring"]
    )

    explicit_metal = (
        re.search(r"\bgold\b", query_lower)
        or re.search(r"\bsilver\b", query_lower)
    )

    previous_filters = state["filters"]

    if explicit_category or explicit_metal:
        previous_filters = {
            "category": None,
            "metal": None,
            "karat": None,
            "min_price": None,
            "max_price": None
        }

    state["filters"] = extract_filters(
        query,
        previous_filters
    )

    # Rebuild complete result set.
    state["result_set"] = create_result_set(
        query,
        state["filters"]
    )

    # Start from page 1.
    state["page"] = 1

    # Clear previous references.
    state["last_displayed_ids"] = []
    state["last_referenced_product_id"] = None

    display_state()

    if not state["result_set"]:
        print("\n===== FINAL ANSWER =====")
        print(
            "I couldn't find any products matching "
            "your requirements."
        )
        return

    displayed = get_current_page()

    display_products(
        displayed,
        "FINAL RETRIEVED PRODUCTS"
    )

    print("\n===== GENERATING ANSWER =====")

    try:
        answer = generate_answer(
            query,
            displayed
        )

        print("\n===== GENERATED ANSWER =====")

        print(answer)

        print("\n===== ANSWER VALIDATION =====")

        if validate_answer(answer, displayed):
            print("VALIDATION: PASSED")
            print("\n===== FINAL ANSWER =====")
            print(answer)
        else:
            print("VALIDATION: FAILED")
            print("\nUsing safe catalog fallback...")

            print("\n===== FINAL ANSWER =====")
            print(fallback_answer(displayed))

    except Exception as e:
        print("\nLLM ERROR:")
        print(e)

        print("\n===== FINAL ANSWER =====")
        print(fallback_answer(displayed))


# ============================================================
# 24. MAIN LOOP
# ============================================================

print("\n============================================")
print("     PROJECT 27 - PRODUCT RESULT SET")
print("============================================")

print("\nJewellery shopping assistant")
print("Type 'exit' to quit.")
print("Type 'show more' for the next page.")
print("Type 'state' to see the current search state.")
print("Type 'reset' to clear the current search.")


while True:

    query = input(
        "\nAsk your jewellery question: "
    ).strip()

    if not query:
        continue

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    if query.lower() == "state":
        display_state()
        continue

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if query.lower() == "reset":
        state["filters"] = {
            "category": None,
            "metal": None,
            "karat": None,
            "min_price": None,
            "max_price": None
        }

        state["result_set"] = []
        state["page"] = 1
        state["last_displayed_ids"] = []
        state["last_referenced_product_id"] = None

        print("Search state reset.")
        continue

    # --------------------------------------------------------
    # SHOW MORE
    # --------------------------------------------------------

    if is_show_more(query):

        if not state["result_set"]:
            print(
                "\nThere is no active product search. "
                "Please search for products first."
            )
            continue

        next_products = show_next_page()

        if not next_products:
            print(
                "\nThere are no more products in "
                "the current result set."
            )
            continue

        display_products(
            next_products,
            "NEXT PRODUCT PAGE"
        )

        continue

    # --------------------------------------------------------
    # CHEAPEST
    # --------------------------------------------------------

    if is_cheapest_question(query):

        if not state["result_set"]:
            print(
                "\nPlease search for some products first."
            )
            continue

        print("\n===== FINAL ANSWER =====")
        print(answer_cheapest())

        continue

    # --------------------------------------------------------
    # MOST EXPENSIVE
    # --------------------------------------------------------

    if is_most_expensive_question(query):

        if not state["result_set"]:
            print(
                "\nPlease search for some products first."
            )
            continue

        print("\n===== FINAL ANSWER =====")
        print(answer_most_expensive())

        continue

    # --------------------------------------------------------
    # ORDINAL PRODUCT FOLLOW-UP
    # --------------------------------------------------------

    ordinal_match = re.search(
        r"(?:tell me more about|details about|information about|"
        r"more about)\s+(?:the\s+)?"
        r"(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)"
        r"(?:\s+one|\s+product)?",
        query.lower()
    )

    if ordinal_match and state["result_set"]:

        identifier = ordinal_match.group(1)

        product = find_product_in_current_results(
            identifier
        )

        if product:
            state["last_referenced_product_id"] = product["id"]

            print("\n===== REFERENCED PRODUCT =====")
            print(f"ID: {product['id']}")
            print(f"Name: {product['name']}")
            print(f"Category: {product['category']}")
            print(f"Metal: {product['metal']}")
            print(f"Karat: {product['karat']}")
            print(f"Price: ₹{product['price']:,}")
            print(f"Description: {product['description']}")

            continue

    # --------------------------------------------------------
    # OTHERWISE:
    # NEW / UPDATED PRODUCT SEARCH
    # --------------------------------------------------------

    perform_new_product_search(query)
