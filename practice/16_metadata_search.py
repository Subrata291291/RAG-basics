# ============================================================
# 16_metadata_search.py
#
# NATURAL LANGUAGE QUERY
#        ↓
# FILTER EXTRACTION
#        ↓
# METADATA FILTERING
#        ↓
# MATCHING PRODUCTS
# ============================================================

import re


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
        "price": 18000
    },

    {
        "id": "R002",
        "name": "Elegant 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 22,
        "price": 25000
    },

    {
        "id": "R003",
        "name": "Diamond Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 18,
        "price": 19500
    },

    {
        "id": "R004",
        "name": "Silver Ring",
        "category": "ring",
        "metal": "silver",
        "karat": 0,
        "price": 8000
    },

    {
        "id": "N001",
        "name": "22K Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": 22,
        "price": 35000
    },

    {
        "id": "N002",
        "name": "Diamond Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": 18,
        "price": 45000
    },

    {
        "id": "B001",
        "name": "22K Gold Bracelet",
        "category": "bracelet",
        "metal": "gold",
        "karat": 22,
        "price": 22000
    },

    {
        "id": "E001",
        "name": "Gold Earrings",
        "category": "earrings",
        "metal": "gold",
        "karat": 22,
        "price": 15000
    }
]


# ============================================================
# 2. EXTRACT FILTERS
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

    max_price_patterns = [

        r"\bunder\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bbelow\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bless\s+than\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bupto\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bup\s+to\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?"
    ]


    for pattern in max_price_patterns:

        match = re.search(
            pattern,
            query
        )

        if match:

            value = float(
                match.group(1).replace(",", "")
            )

            unit = match.group(2)

            if unit:

                value *= 1000

            filters["max_price"] = int(value)

            break


    # ========================================================
    # MIN PRICE
    # ========================================================

    min_price_patterns = [

        r"\babove\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bover\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?",

        r"\bmore\s+than\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k)?"
    ]


    for pattern in min_price_patterns:

        match = re.search(
            pattern,
            query
        )

        if match:

            value = float(
                match.group(1).replace(",", "")
            )

            unit = match.group(2)

            if unit:

                value *= 1000

            filters["min_price"] = int(value)

            break


    return filters


# ============================================================
# 3. FILTER PRODUCTS
# ============================================================

def filter_products(products, filters):

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
        # PRODUCT PASSED ALL FILTERS
        # ----------------------------------------------------

        results.append(product)


    return results


# ============================================================
# 4. DISPLAY PRODUCTS
# ============================================================

def display_products(results):

    if not results:

        print(
            "\nNo matching products found."
        )

        return


    print(
        f"\n===== {len(results)} MATCHING PRODUCTS ====="
    )


    for product in results:

        print(
            f"\n{product['id']} - "
            f"{product['name']}"
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


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

print("\n========================================")
print("       JEWELLERY METADATA SEARCH")
print("========================================")

print(
    "\nTry:"
)

print(
    "show me 22k gold rings under 20k"
)

print(
    "show me gold necklaces above 30k"
)

print(
    "show me silver rings below 10000"
)

print(
    "show me 22k gold bracelets"
)

print(
    "\nType 'exit' to quit."
)


# ============================================================
# 6. CHAT LOOP
# ============================================================

while True:

    query = input(
        "\nAsk for jewellery: "
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
    # EXTRACT FILTERS
    # ========================================================

    filters = extract_filters(
        query
    )


    # ========================================================
    # SHOW FILTERS
    # ========================================================

    print(
        "\n===== EXTRACTED FILTERS ====="
    )


    for key, value in filters.items():

        print(
            f"{key:12} → {value}"
        )


    # ========================================================
    # FILTER PRODUCTS
    # ========================================================

    results = filter_products(
        products,
        filters
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    display_products(
        results
    )