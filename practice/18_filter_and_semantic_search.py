# ============================================================
# 18_filter_and_semantic_search.py
#
# METADATA FILTERING
# +
# SEMANTIC SEARCH
#
# Structured requirements are handled exactly.
# Descriptive requirements are handled semantically.
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
            "An elegant 22K gold ring with a beautiful traditional design, "
            "suitable for weddings and special occasions."
    },

    {
        "id": "R003",
        "name": "Diamond Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 18,
        "price": 19500,
        "description":
            "An 18K gold diamond ring with a modern design, "
            "ideal for special occasions and weddings."
    },

    {
        "id": "R004",
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
            "A 22K gold bracelet with a simple traditional design "
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
# 3. CREATE PRODUCT TEXT
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
# 4. CREATE PRODUCT EMBEDDINGS
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
# 5. PRICE CONVERSION
# ============================================================

def parse_price(
    value,
    unit=None
):

    value = value.replace(
        ",",
        ""
    )

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

    if re.search(
        r"\brings?\b",
        query
    ):

        filters["category"] = "ring"


    elif re.search(
        r"\bnecklaces?\b",
        query
    ):

        filters["category"] = "necklace"


    elif re.search(
        r"\bbracelets?\b",
        query
    ):

        filters["category"] = "bracelet"


    elif re.search(
        r"\b(?:earrings?|ear\s*rings?)\b",
        query
    ):

        filters["category"] = "earrings"


    # ========================================================
    # METAL
    # ========================================================

    if re.search(
        r"\bgold\b",
        query
    ):

        filters["metal"] = "gold"


    elif re.search(
        r"\bsilver\b",
        query
    ):

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
# 7. METADATA FILTERING
# ============================================================

def metadata_filter(filters):

    results = []


    for product in products:


        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        if filters["category"] is not None:

            if product["category"] != filters["category"]:

                continue


        # ----------------------------------------------------
        # METAL
        # ----------------------------------------------------

        if filters["metal"] is not None:

            if product["metal"] != filters["metal"]:

                continue


        # ----------------------------------------------------
        # KARAT
        # ----------------------------------------------------

        if filters["karat"] is not None:

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
        # PRODUCT PASSED FILTERS
        # ----------------------------------------------------

        results.append(product)


    return results


# ============================================================
# 8. SEMANTIC RANKING
# ============================================================

def semantic_rank(
    query,
    filtered_products
):

    if not filtered_products:

        return []


    # --------------------------------------------------------
    # Create embeddings only for filtered products
    # --------------------------------------------------------

    filtered_texts = [

        product_to_text(product)

        for product in filtered_products
    ]


    filtered_embeddings = model.encode(
        filtered_texts
    )


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        query
    )


    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(

        [query_embedding],

        filtered_embeddings

    )[0]


    # --------------------------------------------------------
    # Combine product + score
    # --------------------------------------------------------

    results = []


    for index, product in enumerate(
        filtered_products
    ):

        results.append({

            "product": product,

            "score": similarities[index]
        })


    # --------------------------------------------------------
    # Sort highest similarity first
    # --------------------------------------------------------

    results.sort(

        key=lambda item: item["score"],

        reverse=True
    )


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
# 10. DISPLAY RESULTS
# ============================================================

def display_results(results):

    if not results:

        print(
            "\nNo matching products found."
        )

        return


    print(
        "\n===== SEMANTICALLY RANKED PRODUCTS ====="
    )


    for rank, item in enumerate(
        results,
        start=1
    ):

        product = item["product"]

        score = item["score"]


        print(
            f"\nRank {rank}"
        )

        print(
            f"Semantic Score: {score:.3f}"
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
# 11. MAIN SEARCH
# ============================================================

def search(query):

    print(
        "\n========================================"
    )

    print(
        "USER QUERY"
    )

    print(
        "========================================"
    )

    print(query)


    # ========================================================
    # EXTRACT FILTERS
    # ========================================================

    filters = extract_filters(
        query
    )


    display_filters(
        filters
    )


    # ========================================================
    # METADATA FILTER
    # ========================================================

    filtered_products = metadata_filter(
        filters
    )


    print(
        "\n===== AFTER METADATA FILTER ====="
    )


    print(
        f"Products remaining: "
        f"{len(filtered_products)}"
    )


    for product in filtered_products:

        print(
            f"- {product['name']} "
            f"| ₹{product['price']:,}"
        )


    # ========================================================
    # SEMANTIC RANKING
    # ========================================================

    results = semantic_rank(

        query,

        filtered_products
    )


    display_results(
        results
    )


# ============================================================
# 12. TEST QUERIES
# ============================================================

print(
    "\n========================================"
)

print(
    "       FILTER + SEMANTIC SEARCH"
)

print(
    "========================================"
)


search(
    "show me a beautiful 22k gold ring "
    "under 30k for a wedding"
)


search(
    "I want a modern gold ring under 20k"
)


search(
    "show me something elegant for a wedding"
)


# ============================================================
# 13. INTERACTIVE LOOP
# ============================================================

print(
    "\n========================================"
)

print(
    "          JEWELLERY SEARCH"
)

print(
    "========================================"
)

print(
    "\nTry:"
)

print(
    "show me a beautiful 22k gold ring "
    "under 30k for a wedding"
)

print(
    "I want a modern gold ring under 20k"
)

print(
    "show me something elegant for a wedding"
)

print(
    "\nType 'exit' to quit."
)


while True:

    query = input(
        "\nAsk for jewellery: "
    ).strip()


    # ========================================================
    # EXIT
    # ========================================================

    if query.lower() == "exit":

        print(
            "\nGoodbye!"
        )

        break


    # ========================================================
    # EMPTY
    # ========================================================

    if not query:

        print(
            "Please enter a query."
        )

        continue


    # ========================================================
    # SEARCH
    # ========================================================

    search(
        query
    )