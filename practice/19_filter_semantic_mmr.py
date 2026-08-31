# ============================================================
# 19_filter_semantic_mmr.py
#
# METADATA FILTER
# +
# SEMANTIC SEARCH
# +
# MMR
#
# Structured requirements:
#   category, metal, karat, price
#
# Semantic requirements:
#   beautiful, elegant, wedding, traditional, modern...
#
# MMR:
#   Keeps results relevant while reducing redundancy.
# ============================================================


import re

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PRODUCT DATABASE
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
            "An elegant 22K gold ring with a beautiful traditional "
            "design, suitable for weddings and special occasions."
    },

    {
        "id": "R003",
        "name": "Traditional 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 22,
        "price": 27000,
        "description":
            "A traditional 22K gold ring featuring an intricate "
            "Indian design, perfect for weddings and festive occasions."
    },

    {
        "id": "R004",
        "name": "Bridal 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 22,
        "price": 29000,
        "description":
            "A beautiful bridal 22K gold ring with an elaborate "
            "traditional design made for wedding ceremonies."
    },

    {
        "id": "R005",
        "name": "Modern 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 22,
        "price": 24000,
        "description":
            "A modern 22K gold ring with a minimal contemporary "
            "design suitable for parties and special occasions."
    },

    {
        "id": "R006",
        "name": "Diamond Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 18,
        "price": 19500,
        "description":
            "An 18K gold diamond ring with a modern design, "
            "ideal for special occasions and weddings."
    },

    {
        "id": "R007",
        "name": "Silver Ring",
        "category": "ring",
        "metal": "silver",
        "karat": 0,
        "price": 8000,
        "description":
            "A stylish silver ring with a simple modern design."
    },

    {
        "id": "N001",
        "name": "22K Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": 22,
        "price": 35000,
        "description":
            "A beautiful 22K gold necklace suitable for festive "
            "occasions and weddings."
    },

    {
        "id": "N002",
        "name": "Diamond Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": 18,
        "price": 45000,
        "description":
            "An 18K gold necklace featuring diamond detailing "
            "for weddings and special occasions."
    },

    {
        "id": "B001",
        "name": "22K Gold Bracelet",
        "category": "bracelet",
        "metal": "gold",
        "karat": 22,
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
        "karat": 22,
        "price": 15000,
        "description":
            "22K gold earrings with an elegant design suitable "
            "for weddings and festive occasions."
    }
]


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 3. PRODUCT TEXT
# ============================================================

def product_to_text(product):

    return f"""
{product['name']}

Category: {product['category']}

Metal: {product['metal']}

Karat: {product['karat']}K

Price: ₹{product['price']}

Description:
{product['description']}
"""


# ============================================================
# 4. CREATE ALL PRODUCT EMBEDDINGS
# ============================================================

print("\nCreating product embeddings...")

product_texts = [

    product_to_text(product)

    for product in products
]


product_embeddings = model.encode(
    product_texts
)

print("Product embeddings created.")


# ============================================================
# 5. PARSE PRICE
# ============================================================

def parse_price(value, unit=None):

    value = value.replace(",", "")

    number = float(value)

    if unit:

        if unit.lower() == "k":

            number *= 1000

    return int(number)


# ============================================================
# 6. EXTRACT FILTERS
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
# 7. METADATA FILTER
# ============================================================

def metadata_filter(filters):

    results = []


    for product in products:


        if filters["category"] is not None:

            if product["category"] != filters["category"]:

                continue


        if filters["metal"] is not None:

            if product["metal"] != filters["metal"]:

                continue


        if filters["karat"] is not None:

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
# 8. MMR SEARCH
# ============================================================

def mmr_search(
    query,
    filtered_products,
    top_k=3,
    lambda_value=0.7
):

    if not filtered_products:

        return []


    # --------------------------------------------------------
    # Get embeddings for filtered products
    # --------------------------------------------------------

    filtered_indices = [

        products.index(product)

        for product in filtered_products
    ]


    candidate_embeddings = product_embeddings[
        filtered_indices
    ]


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        query
    )


    # --------------------------------------------------------
    # Query similarity
    # --------------------------------------------------------

    query_scores = cosine_similarity(

        [query_embedding],

        candidate_embeddings

    )[0]


    # --------------------------------------------------------
    # Document-to-document similarity
    # --------------------------------------------------------

    document_similarities = cosine_similarity(

        candidate_embeddings

    )


    # --------------------------------------------------------
    # MMR selection
    # --------------------------------------------------------

    selected = []

    remaining = list(
        range(len(filtered_products))
    )


    while remaining and len(selected) < top_k:

        best_index = None

        best_mmr_score = float("-inf")


        for candidate in remaining:


            # =================================================
            # RELEVANCE
            # =================================================

            relevance = query_scores[candidate]


            # =================================================
            # REDUNDANCY
            # =================================================

            if not selected:

                redundancy = 0

            else:

                redundancy = max(

                    document_similarities[
                        candidate
                    ][selected]
                )


            # =================================================
            # MMR
            # =================================================

            mmr_score = (

                lambda_value * relevance

                -

                (1 - lambda_value) * redundancy
            )


            # =================================================
            # FIND BEST CANDIDATE
            # =================================================

            if mmr_score > best_mmr_score:

                best_mmr_score = mmr_score

                best_index = candidate


        # =====================================================
        # SELECT
        # =====================================================

        selected.append(
            best_index
        )

        remaining.remove(
            best_index
        )


    # ========================================================
    # CREATE RESULTS
    # ========================================================

    results = []


    for index in selected:

        results.append({

            "product": filtered_products[index],

            "query_score": query_scores[index],

            "mmr_score": (

                lambda_value * query_scores[index]

                -

                (1 - lambda_value)
                *
                (
                    0
                    if len(
                        selected
                    ) == 1
                    else max(
                        document_similarities[
                            index
                        ][
                            [
                                x
                                for x in selected
                                if x != index
                            ]
                        ]
                    )
                )
            )
        })


    return results


# ============================================================
# 9. DISPLAY FILTERS
# ============================================================

def display_filters(filters):

    print(
        "\n===== DETECTED FILTERS ====="
    )


    for key, value in filters.items():

        print(
            f"{key:12} → {value}"
        )


# ============================================================
# 10. DISPLAY FILTERED PRODUCTS
# ============================================================

def display_filtered_products(
    filtered_products
):

    print(
        "\n===== AFTER METADATA FILTER ====="
    )


    print(
        f"Products remaining: "
        f"{len(filtered_products)}"
    )


    for product in filtered_products:

        print(

            f"- {product['id']} | "
            f"{product['name']} | "
            f"₹{product['price']:,}"
        )


# ============================================================
# 11. DISPLAY MMR RESULTS
# ============================================================

def display_mmr_results(results):

    print(
        "\n===== MMR RESULTS ====="
    )


    for rank, item in enumerate(
        results,
        start=1
    ):

        product = item["product"]


        print(
            f"\nRank {rank}"
        )


        print(
            f"Query Similarity = "
            f"{item['query_score']:.3f}"
        )


        print(
            f"MMR Score = "
            f"{item['mmr_score']:.3f}"
        )


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
            f"Karat: {product['karat']}K"
        )


        print(
            f"Price: ₹{product['price']:,}"
        )


        print(
            f"Description: "
            f"{product['description']}"
        )


# ============================================================
# 12. COMPLETE SEARCH PIPELINE
# ============================================================

def search(query):

    print(
        "\n\n========================================"
    )

    print(
        "USER QUERY"
    )

    print(
        "========================================"
    )

    print(query)


    # ========================================================
    # STEP 1
    # ========================================================

    filters = extract_filters(
        query
    )


    display_filters(
        filters
    )


    # ========================================================
    # STEP 2
    # ========================================================

    filtered_products = metadata_filter(
        filters
    )


    display_filtered_products(
        filtered_products
    )


    if not filtered_products:

        print(
            "\nNo products match the "
            "specified filters."
        )

        return


    # ========================================================
    # STEP 3
    # ========================================================

    results = mmr_search(

        query,

        filtered_products,

        top_k=3,

        lambda_value=0.7
    )


    # ========================================================
    # STEP 4
    # ========================================================

    display_mmr_results(
        results
    )


# ============================================================
# 13. INTERACTIVE LOOP
# ============================================================

print(
    "\n========================================"
)

print(
    "       JEWELLERY MMR SEARCH"
)

print(
    "========================================"
)

print(
    "\nExamples:"
)

print(
    "show me a beautiful 22k gold ring "
    "under 30k for a wedding"
)

print(
    "I want a modern gold ring under 30k"
)

print(
    "show me a traditional gold ring"
)

print(
    "\nType 'exit' to quit."
)


while True:

    query = input(
        "\nAsk for jewellery: "
    ).strip()


    if query.lower() == "exit":

        print(
            "\nGoodbye!"
        )

        break


    if not query:

        print(
            "Please enter a query."
        )

        continue


    search(
        query
    )



"""
                 USER
                   │
                   ▼
            Query Extraction
                   │
                   ▼
          ┌─────────────────┐
          │ Metadata Filter │
          └────────┬────────┘
                   │
             Valid Products
                   │
                   ▼
          Semantic Similarity
                   │
                   ▼
                 MMR
                   │
                   ▼
           Best Diverse Items
                   │
                   ▼
              LLM Answer
"""