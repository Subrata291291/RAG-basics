# src/query_router.py

import re


# ============================================================
# NORMAL CONVERSATION
# ============================================================

NORMAL_CHAT_PATTERNS = [

    # Greetings
    "hi",
    "hello",
    "hey",
    "hii",
    "hiii",

    "good morning",
    "good afternoon",
    "good evening",

    # Thanks
    "thank you",
    "thanks",
    "thank",
    "thankyou",

    # Short conversational responses
    "ok",
    "okay",
    "great",
    "nice",
    "perfect",
    "good",

    # Capability questions
    "what can you do",
    "what do you do",
    "how can you help",
    "who are you"
]


# ============================================================
# NORMALIZE QUERY
# ============================================================

def normalize_query(query):

    if not query:
        return ""

    query = query.lower().strip()

    # Remove extra spaces
    query = re.sub(
        r"\s+",
        " ",
        query
    )

    return query


# ============================================================
# CHECK NORMAL CHAT
# ============================================================

def is_normal_chat(query):

    query = normalize_query(query)

    if not query:
        return False

    # Exact matches
    if query in NORMAL_CHAT_PATTERNS:
        return True

    # Conversational phrases
    for pattern in NORMAL_CHAT_PATTERNS:

        if (
            query.startswith(pattern + " ")
            or query.endswith(" " + pattern)
        ):
            return True

    return False


# ============================================================
# KNOWLEDGE QUERY DETECTION
# ============================================================

KNOWLEDGE_KEYWORDS = [

    # Policies
    "return",
    "returns",
    "refund",
    "refunds",

    "shipping",
    "delivery",
    "deliver",

    "privacy",
    "personal information",

    "terms",
    "conditions",
    "terms and conditions",

    # Store information
    "policy",
    "policies"
]


def is_knowledge_query(query):

    query = normalize_query(query)

    for keyword in KNOWLEDGE_KEYWORDS:

        if keyword in query:

            return True

    return False


# ============================================================
# PRODUCT QUERY DETECTION
# ============================================================

PRODUCT_KEYWORDS = [

    # Product categories
    "ring",
    "rings",

    "necklace",
    "necklaces",

    "bracelet",
    "bracelets",

    "earring",
    "earrings",

    # Jewellery terms
    "jewellery",
    "jewelry",

    # Metals
    "gold",
    "silver",

    # Karat
    "18k",
    "18 k",

    "22k",
    "22 k",

    "24k",
    "24 k",

    # Buying/search terms
    "buy",
    "purchase",
    "show me",
    "find me",
    "looking for",

    # Price-related
    "under",
    "below",
    "above",
    "over",
    "cheap",
    "cheapest",
    "expensive",
    "budget",

    "price",
    "priced"
]


def is_product_query(query):

    query = normalize_query(query)

    for keyword in PRODUCT_KEYWORDS:

        if keyword in query:

            return True

    return False


# ============================================================
# FOLLOW-UP QUERY DETECTION
# ============================================================

FOLLOWUP_PATTERNS = [

    "which one",
    "which is",
    "which one is",
    "what about",
    "how about",

    "cheapest",
    "most expensive",

    "this one",
    "that one",

    "the first one",
    "the second one",
    "the third one",

    "tell me more",
    "more details",

    "show more",

    "compare them",
    "compare these",

    "is it available",
    "is this available",

    "how much is it",

    "what is the price",

    "what's the price",

    "karat",
    "metal",

    "details"
]


def is_followup(query):

    query = normalize_query(query)

    for pattern in FOLLOWUP_PATTERNS:

        if pattern in query:

            return True

    return False


# ============================================================
# CLASSIFY QUERY
# ============================================================

def classify_query(
    query,
    has_previous_products=False
):
    """
    Determine what type of request the user made.

    Possible results:

        normal
        product
        knowledge
        followup
    """

    query = normalize_query(query)


    # --------------------------------------------------------
    # Empty query
    # --------------------------------------------------------

    if not query:

        return "normal"


    # --------------------------------------------------------
    # Follow-up gets priority when products exist
    # --------------------------------------------------------

    if (
        has_previous_products
        and is_followup(query)
    ):

        return "followup"


    # --------------------------------------------------------
    # Normal conversation
    # --------------------------------------------------------

    if is_normal_chat(query):

        return "normal"


    # --------------------------------------------------------
    # Knowledge / policy
    # --------------------------------------------------------

    if is_knowledge_query(query):

        return "knowledge"


    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    if is_product_query(query):

        return "product"


    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return "normal"