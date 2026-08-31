import os
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PRODUCTS_FILE = DATA_DIR / "products.json"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
PROMPTS_DIR = BASE_DIR / "prompts"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

MAX_PRODUCT_RESULTS = 10
SEMANTIC_TOP_K = 5

MAX_CONVERSATION_TURNS = 8


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found in .env"
    )


# ============================================================
# OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ============================================================
# LOAD PRODUCTS
# ============================================================

def load_products() -> List[Dict[str, Any]]:

    if not PRODUCTS_FILE.exists():
        raise FileNotFoundError(
            f"Products file not found:\n{PRODUCTS_FILE}"
        )

    with open(
        PRODUCTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        products = json.load(file)

    if not isinstance(products, list):
        raise ValueError(
            "products.json must contain a JSON list."
        )

    required_fields = {
        "id",
        "name",
        "category",
        "metal",
        "karat",
        "price",
        "description"
    }

    for product in products:

        missing = required_fields - set(product.keys())

        if missing:
            raise ValueError(
                f"Product {product.get('id')} "
                f"is missing fields: {missing}"
            )

    return products


products = load_products()


# ============================================================
# LOAD KNOWLEDGE DOCUMENTS
# ============================================================

def load_knowledge_documents():

    documents = []

    if not KNOWLEDGE_DIR.exists():
        print(
            f"WARNING: Knowledge directory not found: "
            f"{KNOWLEDGE_DIR}"
        )
        return documents

    for file_path in KNOWLEDGE_DIR.glob("*.txt"):

        try:

            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if text:

                documents.append(
                    {
                        "name": file_path.name,
                        "text": text
                    }
                )

        except Exception as e:

            print(
                f"WARNING: Could not read "
                f"{file_path.name}: {e}"
            )

    return documents


knowledge_documents = load_knowledge_documents()


# ============================================================
# LOAD PROMPTS
# ============================================================

def load_prompt(filename: str) -> str:

    path = PROMPTS_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"""
Required prompt file not found:

{path}

Create this file inside the prompts/ folder.
"""
        )

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:

        raise ValueError(
            f"Prompt file is empty: {path}"
        )

    return text


# ============================================================
# PROMPT LOADING
# ============================================================

PROMPTS = {

    "intent":
        load_prompt(
            "intent_classification.txt"
        ),

    "filters":
        load_prompt(
            "filter_extraction.txt"
        ),

    "product_answer":
        load_prompt(
            "product_answer.txt"
        ),

    "product_followup":
        load_prompt(
            "product_followup.txt"
        ),

    "knowledge_answer":
        load_prompt(
            "knowledge_answer.txt"
        ),

    "safety":
        load_prompt(
            "safety.txt"
        )
}


# ============================================================
# EMBEDDING MODEL
# ============================================================

print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print(
    "Embedding model loaded."
)


# ============================================================
# PRODUCT EMBEDDINGS
# ============================================================

def create_product_text(product):

    return (
        f"{product['name']}. "
        f"Category: {product['category']}. "
        f"Metal: {product['metal']}. "
        f"Karat: {product['karat']}. "
        f"Price: {product['price']}. "
        f"{product['description']}"
    )


product_texts = [
    create_product_text(product)
    for product in products
]


print(
    "Creating product embeddings..."
)

product_embeddings = embedding_model.encode(
    product_texts
)

print(
    "Product embeddings created."
)


# ============================================================
# CONVERSATION STATE
# ============================================================

conversation_history = []

current_filters = {

    "category": None,
    "metal": None,
    "karat": None,
    "min_price": None,
    "max_price": None
}

current_results = []

current_page = 1


# ============================================================
# UTILITY: NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    return (
        text
        .lower()
        .strip()
    )


# ============================================================
# UTILITY: EXTRACT JSON FROM LLM
# ============================================================

def extract_json(text):

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)

    except Exception:
        pass

    # JSON inside markdown
    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        re.DOTALL
    )

    if match:

        try:
            return json.loads(
                match.group(1)
            )

        except Exception:
            pass

    # First JSON object
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        try:
            return json.loads(
                match.group(0)
            )

        except Exception:
            pass

    return None


# ============================================================
# LLM CALL
# ============================================================

def ask_llm(
    system_prompt,
    user_prompt,
    temperature=0
):

    response = client.chat.completions.create(

        model="openrouter/free",

        temperature=temperature,

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(query):

    prompt = f"""
{PROMPTS["intent"]}

CUSTOMER QUERY:
{query}

Return ONLY valid JSON.

Example:

{{
    "intent": "product_search"
}}

Allowed intents:

product_search
product_followup
knowledge
unknown
"""

    raw = ask_llm(
        PROMPTS["safety"],
        prompt
    )

    result = extract_json(raw)

    if not result:

        return "unknown"

    intent = result.get(
        "intent",
        "unknown"
    )

    allowed = {

        "product_search",
        "product_followup",
        "knowledge",
        "unknown"
    }

    if intent not in allowed:

        return "unknown"

    return intent


# ============================================================
# FILTER EXTRACTION
# ============================================================

def extract_filters(query):

    prompt = f"""
{PROMPTS["filters"]}

CUSTOMER QUERY:
{query}

Return ONLY JSON.

Required structure:

{{
    "category": null,
    "metal": null,
    "karat": null,
    "min_price": null,
    "max_price": null
}}

Important:

- category should be one of:
  ring, necklace, bracelet, earrings

- metal should be one of:
  gold, silver, platinum

- karat should be a string such as:
  18K, 22K, 24K

- prices must be numbers only.

- Do not guess missing information.
- Use null when information is not present.
"""

    raw = ask_llm(
        PROMPTS["filters"],
        prompt
    )

    result = extract_json(raw)

    if not result:

        return {
            "category": None,
            "metal": None,
            "karat": None,
            "min_price": None,
            "max_price": None
        }

    return {

        "category":
            result.get("category"),

        "metal":
            result.get("metal"),

        "karat":
            result.get("karat"),

        "min_price":
            result.get("min_price"),

        "max_price":
            result.get("max_price")
    }


# ============================================================
# CLEAN FILTERS
# ============================================================

def clean_filters(filters):

    cleaned = {}

    for key, value in filters.items():

        if value is None:

            cleaned[key] = None

            continue

        if isinstance(value, str):

            value = value.strip()

            if value.lower() in {
                "",
                "none",
                "null"
            }:

                cleaned[key] = None

                continue

        cleaned[key] = value

    return cleaned


# ============================================================
# APPLY FILTERS
# ============================================================

def filter_products(filters):

    results = []

    for product in products:

        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        if filters["category"]:

            if normalize_text(
                product["category"]
            ) != normalize_text(
                filters["category"]
            ):

                continue

        # ----------------------------------------------------
        # METAL
        # ----------------------------------------------------

        if filters["metal"]:

            if normalize_text(
                product["metal"]
            ) != normalize_text(
                filters["metal"]
            ):

                continue

        # ----------------------------------------------------
        # KARAT
        # ----------------------------------------------------

        if filters["karat"]:

            if normalize_text(
                product["karat"]
            ) != normalize_text(
                filters["karat"]
            ):

                continue

        # ----------------------------------------------------
        # MIN PRICE
        # ----------------------------------------------------

        if filters["min_price"] is not None:

            if product["price"] < float(
                filters["min_price"]
            ):

                continue

        # ----------------------------------------------------
        # MAX PRICE
        # ----------------------------------------------------

        if filters["max_price"] is not None:

            if product["price"] > float(
                filters["max_price"]
            ):

                continue

        results.append(product)

    return results


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    candidate_products,
    top_k=SEMANTIC_TOP_K
):

    if not candidate_products:

        return []

    candidate_indices = []

    for product in candidate_products:

        for index, original in enumerate(products):

            if original["id"] == product["id"]:

                candidate_indices.append(index)

                break

    query_embedding = embedding_model.encode(
        query
    )

    similarities = cosine_similarity(
        [query_embedding],
        product_embeddings[
            candidate_indices
        ]
    )[0]

    ranked = []

    for position, score in enumerate(
        similarities
    ):

        ranked.append(
            {
                "product":
                    candidate_products[position],

                "score":
                    float(score)
            }
        )

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:top_k]


# ============================================================
# DISPLAY FILTERS
# ============================================================

def display_filters(filters):

    print(
        "\n===== CURRENT SEARCH STATE ====="
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
        f"min_price -> {filters['min_price']}"
    )

    print(
        f"max_price -> {filters['max_price']}"
    )


# ============================================================
# DISPLAY PRODUCTS
# ============================================================

def display_products(results):

    if not results:

        print(
            "\nNo matching products found."
        )

        return

    print(
        "\n===== CURRENT RESULT SET ====="
    )

    print(
        f"Showing {len(results)} product(s)."
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        product = (
            result["product"]
            if isinstance(result, dict)
            and "product" in result
            else result
        )

        score = (
            result.get("score")
            if isinstance(result, dict)
            else None
        )

        print(
            f"\n{index}. {product['name']}"
        )

        print(
            f"   ID: {product['id']}"
        )

        print(
            f"   Category: {product['category']}"
        )

        print(
            f"   Metal: {product['metal']}"
        )

        print(
            f"   Karat: {product['karat']}"
        )

        print(
            f"   Price: ₹{product['price']:,}"
        )

        if score is not None:

            print(
                f"   Semantic Score: {score:.3f}"
            )

        print(
            f"   {product['description']}"
        )


# ============================================================
# RESULT SET OPERATIONS
# ============================================================

def cheapest_product():

    if not current_results:

        return None

    return min(
        current_results,
        key=lambda x:
            x["product"]["price"]
    )


def most_expensive_product():

    if not current_results:

        return None

    return max(
        current_results,
        key=lambda x:
            x["product"]["price"]
    )


def get_product_by_position(position):

    if not current_results:

        return None

    index = position - 1

    if (
        index < 0
        or index >= len(current_results)
    ):

        return None

    return current_results[index]


# ============================================================
# DETECT DETERMINISTIC FOLLOW-UP
# ============================================================

def detect_result_operation(query):

    text = normalize_text(query)

    # --------------------------------------------------------
    # CHEAPEST
    # --------------------------------------------------------

    if (
        "cheapest" in text
        or "lowest price" in text
        or "least expensive" in text
    ):

        return "cheapest"

    # --------------------------------------------------------
    # MOST EXPENSIVE
    # --------------------------------------------------------

    if (
        "most expensive" in text
        or "highest price" in text
        or "costliest" in text
    ):

        return "most_expensive"

    # --------------------------------------------------------
    # FIRST
    # --------------------------------------------------------

    if re.search(
        r"\b(first|1st)\b",
        text
    ):

        return "position:1"

    # --------------------------------------------------------
    # SECOND
    # --------------------------------------------------------

    if re.search(
        r"\b(second|2nd)\b",
        text
    ):

        return "position:2"

    # --------------------------------------------------------
    # THIRD
    # --------------------------------------------------------

    if re.search(
        r"\b(third|3rd)\b",
        text
    ):

        return "position:3"

    # --------------------------------------------------------
    # FOURTH
    # --------------------------------------------------------

    if re.search(
        r"\b(fourth|4th)\b",
        text
    ):

        return "position:4"

    # --------------------------------------------------------
    # FIFTH
    # --------------------------------------------------------

    if re.search(
        r"\b(fifth|5th)\b",
        text
    ):

        return "position:5"

    return None


# ============================================================
# HANDLE RESULT OPERATION
# ============================================================

def handle_result_operation(
    query,
    operation
):

    if operation == "cheapest":

        result = cheapest_product()

        if not result:

            return (
                "There are no matching products "
                "in the current result set."
            )

    elif operation == "most_expensive":

        result = most_expensive_product()

        if not result:

            return (
                "There are no matching products "
                "in the current result set."
            )

    elif operation.startswith(
        "position:"
    ):

        position = int(
            operation.split(":")[1]
        )

        result = get_product_by_position(
            position
        )

        if not result:

            return (
                f"There is no product at "
                f"position {position} "
                f"in the current result set."
            )

    else:

        return None

    product = result["product"]

    return (
        f"{product['name']} "
        f"({product['id']}) is the relevant "
        f"product from the current results.\n\n"
        f"Price: ₹{product['price']:,}\n"
        f"Metal: {product['metal']}\n"
        f"Karat: {product['karat']}\n"
        f"Description: {product['description']}"
    )


# ============================================================
# CREATE PRODUCT CONTEXT
# ============================================================

def create_product_context(
    results
):

    if not results:

        return "No products are currently available."

    lines = []

    for index, result in enumerate(
        results,
        start=1
    ):

        product = result["product"]

        lines.append(
            f"""
Product {index}
ID: {product['id']}
Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}
Price: ₹{product['price']:,}
Description: {product['description']}
"""
        )

    return "\n".join(lines)


# ============================================================
# PRODUCT ANSWER
# ============================================================

def generate_product_answer(
    query,
    results
):

    context = create_product_context(
        results
    )

    prompt = f"""
{PROMPTS["product_answer"]}

CUSTOMER QUESTION:
{query}

PRODUCT CONTEXT:
{context}

Use ONLY the products in the context.

Do not invent products, prices, karats,
features, availability or specifications.
"""

    return ask_llm(
        PROMPTS["product_answer"],
        prompt
    )


# ============================================================
# KNOWLEDGE SEARCH
# ============================================================

def knowledge_search(
    query,
    top_k=3
):

    if not knowledge_documents:

        return []

    document_texts = [
        item["text"]
        for item in knowledge_documents
    ]

    embeddings = embedding_model.encode(
        document_texts
    )

    query_embedding = embedding_model.encode(
        query
    )

    similarities = cosine_similarity(
        [query_embedding],
        embeddings
    )[0]

    indices = similarities.argsort()[
        ::-1
    ][:top_k]

    results = []

    for index in indices:

        results.append(
            {
                "name":
                    knowledge_documents[
                        index
                    ]["name"],

                "text":
                    knowledge_documents[
                        index
                    ]["text"],

                "score":
                    float(
                        similarities[index]
                    )
            }
        )

    return results


# ============================================================
# CREATE KNOWLEDGE CONTEXT
# ============================================================

def create_knowledge_context(
    results
):

    if not results:

        return (
            "No relevant knowledge documents "
            "were found."
        )

    parts = []

    for result in results:

        parts.append(
            f"""
DOCUMENT: {result['name']}

{result['text']}
"""
        )

    return "\n".join(parts)


# ============================================================
# KNOWLEDGE ANSWER
# ============================================================

def generate_knowledge_answer(
    query,
    results
):

    context = create_knowledge_context(
        results
    )

    prompt = f"""
{PROMPTS["knowledge_answer"]}

CUSTOMER QUESTION:
{query}

DOCUMENT CONTEXT:
{context}

Use ONLY the document context.

Never invent:
- policies
- return conditions
- refund times
- shipping times
- privacy practices
- terms
"""

    return ask_llm(
        PROMPTS["knowledge_answer"],
        prompt
    )


# ============================================================
# PRODUCT FOLLOW-UP
# ============================================================

def generate_product_followup(
    query,
    results
):

    context = create_product_context(
        results
    )

    prompt = f"""
{PROMPTS["product_followup"]}

CUSTOMER FOLLOW-UP:
{query}

CURRENT PRODUCT RESULT SET:
{context}

Use ONLY these products.

Every price, product ID, karat,
metal and feature must exactly match
the context.
"""

    return ask_llm(
        PROMPTS["product_followup"],
        prompt
    )


# ============================================================
# REMEMBER CONVERSATION
# ============================================================

def remember_turn(
    user_query,
    assistant_answer
):

    conversation_history.append(
        {
            "user":
                user_query,

            "assistant":
                assistant_answer
        }
    )

    if (
        len(conversation_history)
        > MAX_CONVERSATION_TURNS
    ):

        del conversation_history[
            :-MAX_CONVERSATION_TURNS
        ]


# ============================================================
# MAIN PRODUCT SEARCH
# ============================================================

def perform_product_search(
    query
):

    global current_results
    global current_page
    global current_filters

    extracted = extract_filters(
        query
    )

    extracted = clean_filters(
        extracted
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Only replace filters explicitly supplied
    # in the new query.
    # --------------------------------------------------------

    for key, value in extracted.items():

        if value is not None:

            current_filters[key] = value

    display_filters(
        current_filters
    )

    # --------------------------------------------------------
    # METADATA FILTER
    # --------------------------------------------------------

    candidates = filter_products(
        current_filters
    )

    print(
        "\n===== METADATA FILTER ====="
    )

    print(
        f"Matching products: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # NO MATCHES
    # --------------------------------------------------------

    if not candidates:

        current_results = []
        current_page = 1

        return []

    # --------------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------------

    results = semantic_search(
        query,
        candidates,
        top_k=MAX_PRODUCT_RESULTS
    )

    current_results = results
    current_page = 1

    return results


# ============================================================
# RESET SEARCH STATE
# ============================================================

def reset_search_state():

    global current_filters
    global current_results
    global current_page

    current_filters = {

        "category": None,
        "metal": None,
        "karat": None,
        "min_price": None,
        "max_price": None
    }

    current_results = []

    current_page = 1


# ============================================================
# MAIN CHATBOT
# ============================================================

print()
print("=" * 65)
print(
    "        PROJECT 29 - RESULT-SET-AWARE JEWELLERY RAG"
)
print("=" * 65)

print()
print(
    "Products + Policies + Metadata Filtering"
)
print(
    "Semantic Search + Conversation State"
)
print(
    "Deterministic Result-Set Operations"
)

print()
print("Examples:")

print(
    "  Show me gold rings under ₹30,000"
)

print(
    "  Which one is cheapest?"
)

print(
    "  Tell me about the second one"
)

print(
    "  What about necklaces?"
)

print(
    "  What is your return policy?"
)

print()
print("Commands:")
print("  reset  - clear current search state")
print("  exit   - quit")

print()


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    query = input(
        "Ask your jewellery question: "
    ).strip()

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if query.lower() == "exit":

        print(
            "\nGoodbye!"
        )

        break

    # --------------------------------------------------------
    # EMPTY
    # --------------------------------------------------------

    if not query:

        print(
            "Please enter a question."
        )

        continue

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if query.lower() == "reset":

        reset_search_state()

        print(
            "\nSearch state has been reset."
        )

        continue

    # --------------------------------------------------------
    # INTENT
    # --------------------------------------------------------

    try:

        intent = classify_intent(
            query
        )

    except Exception as e:

        print(
            "\nERROR while classifying intent:"
        )

        print(e)

        continue

    print(
        f"\nQuery type: {intent}"
    )

    # ========================================================
    # PRODUCT FOLLOW-UP
    # ========================================================

    if intent == "product_followup":

        # ----------------------------------------------------
        # FIRST CHECK DETERMINISTIC OPERATIONS
        # ----------------------------------------------------

        operation = detect_result_operation(
            query
        )

        if operation:

            answer = handle_result_operation(
                query,
                operation
            )

            print(
                "\n===== FINAL ANSWER ====="
            )

            print(answer)

            remember_turn(
                query,
                answer
            )

            continue

        # ----------------------------------------------------
        # NO CURRENT RESULT SET
        # ----------------------------------------------------

        if not current_results:

            answer = (
                "There are no products in the "
                "current result set to refer to."
            )

            print(
                "\n===== FINAL ANSWER ====="
            )

            print(answer)

            remember_turn(
                query,
                answer
            )

            continue

        # ----------------------------------------------------
        # LLM FOLLOW-UP
        # ----------------------------------------------------

        print(
            "\n===== USING CURRENT RESULT SET ====="
        )

        display_products(
            current_results
        )

        try:

            answer = generate_product_followup(
                query,
                current_results
            )

        except Exception as e:

            print(
                "\nERROR while generating answer:"
            )

            print(e)

            continue

        print(
            "\n===== FINAL ANSWER ====="
        )

        print(answer)

        remember_turn(
            query,
            answer
        )

        continue

    # ========================================================
    # KNOWLEDGE
    # ========================================================

    if intent == "knowledge":

        print(
            "\n===== KNOWLEDGE SEARCH ====="
        )

        try:

            knowledge_results = knowledge_search(
                query
            )

            for rank, result in enumerate(
                knowledge_results,
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
                    f"Document = "
                    f"{result['name']}"
                )

            if not knowledge_results:

                answer = (
                    "I couldn't find relevant "
                    "information in the available "
                    "knowledge documents."
                )

            else:

                print(
                    "\n===== GENERATING ANSWER ====="
                )

                answer = generate_knowledge_answer(
                    query,
                    knowledge_results
                )

        except Exception as e:

            print(
                "\nERROR:"
            )

            print(e)

            continue

        print(
            "\n===== FINAL ANSWER ====="
        )

        print(answer)

        remember_turn(
            query,
            answer
        )

        continue

    # ========================================================
    # PRODUCT SEARCH
    # ========================================================

    if intent == "product_search":

        try:

            results = perform_product_search(
                query
            )

        except Exception as e:

            print(
                "\nERROR during product search:"
            )

            print(e)

            continue

        if not results:

            answer = (
                "I couldn't find any products "
                "matching your requirements."
            )

            print(
                "\n===== FINAL ANSWER ====="
            )

            print(answer)

            remember_turn(
                query,
                answer
            )

            continue

        print(
            "\n===== SEARCH RESULTS ====="
        )

        display_products(
            results
        )

        print(
            "\n===== GENERATING ANSWER ====="
        )

        try:

            answer = generate_product_answer(
                query,
                results
            )

        except Exception as e:

            print(
                "\nERROR while generating answer:"
            )

            print(e)

            continue

        print(
            "\n===== FINAL ANSWER ====="
        )

        print(answer)

        remember_turn(
            query,
            answer
        )

        continue

    # ========================================================
    # UNKNOWN
    # ========================================================

    answer = (
        "I can help with jewellery products, "
        "orders, returns, shipping, privacy, "
        "terms and other information available "
        "in our website data."
    )

    print(
        "\n===== FINAL ANSWER ====="
    )

    print(answer)

    remember_turn(
        query,
        answer
    )