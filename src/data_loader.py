import json

from src.config import (
    PRODUCTS_FILE,
    KNOWLEDGE_DIR
)


# ============================================================
# LOAD PRODUCTS
# ============================================================

def load_products():

    if not PRODUCTS_FILE.exists():
        raise FileNotFoundError(
            f"Products file not found: {PRODUCTS_FILE}"
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

    return products


# ============================================================
# LOAD KNOWLEDGE DOCUMENTS
# ============================================================

def load_knowledge_documents():

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {KNOWLEDGE_DIR}"
        )

    documents = []

    for file_path in sorted(
        KNOWLEDGE_DIR.glob("*.txt")
    ):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read().strip()

        if not content:
            continue

        documents.append(
            {
                "document": file_path.name,
                "content": content
            }
        )

    return documents


# ============================================================
# LOAD EVERYTHING
# ============================================================

def load_all_data():

    products = load_products()

    knowledge = load_knowledge_documents()

    return {
        "products": products,
        "knowledge": knowledge
    }