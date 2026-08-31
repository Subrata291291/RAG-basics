# ============================================================
# 15_query_filter_extraction.py
# NATURAL LANGUAGE → STRUCTURED JEWELLERY FILTERS
# ============================================================

import re


# ============================================================
# 1. CONVERT PRICE TEXT INTO NUMBER
# ============================================================

def parse_price(value, unit=None):

    value = value.replace(",", "").strip()

    number = float(value)

    # "20k" means ₹20,000 when used as a price
    if unit and unit.lower() == "k":
        number = number * 1000

    return int(number)


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

    elif re.search(r"\b(?:earrings?|ear\s*rings?)\b", query):

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
    #
    # Important:
    #
    # 22k gold → karat = 22
    #
    # But:
    #
    # above 30k → ₹30,000
    #
    # We only treat "22k" as karat when it is
    # associated with gold.
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
    #
    # Examples:
    #
    # under 20000
    # below 20000
    # under 20k
    # less than 20k
    # upto 20000
    # up to 20k
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

            value = match.group(1)

            unit = match.group(2)

            filters["max_price"] = parse_price(
                value,
                unit
            )

            break


    # ========================================================
    # MIN PRICE
    #
    # Examples:
    #
    # above 30000
    # above 30k
    # over 30000
    # over 30k
    # more than 30000
    # more than 30k
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

            value = match.group(1)

            unit = match.group(2)

            filters["min_price"] = parse_price(
                value,
                unit
            )

            break


    return filters


# ============================================================
# 3. DISPLAY FILTERS
# ============================================================

def display_filters(filters):

    print("\n===== EXTRACTED FILTERS =====")

    for key, value in filters.items():

        print(
            f"{key:12} → {value}"
        )


# ============================================================
# 4. TEST QUERIES
# ============================================================

print("\n========================================")
print("        AUTOMATIC TESTS")
print("========================================")


test_queries = [

    "show me 22k gold rings under 20000",

    "show me 22k gold rings under 20k",

    "show me silver rings below ₹10000",

    "show me gold necklaces above ₹30000",

    "show me gold necklaces above 30k",

    "show me 18k gold bracelets under 50k",

    "show me 24k gold earrings above 1 lakh"
]


for query in test_queries:

    print("\n----------------------------------------")

    print("Query:")

    print(query)

    filters = extract_filters(query)

    display_filters(filters)


# ============================================================
# 5. INTERACTIVE CHAT LOOP
# ============================================================

print("\n========================================")
print("     JEWELLERY QUERY FILTER SYSTEM")
print("========================================")

print("\nTry queries such as:")

print(
    "show me 22k gold rings under 20k"
)

print(
    "show me silver rings below 10000"
)

print(
    "show me gold necklaces above 30k"
)

print(
    "show me 18k gold bracelets under 50k"
)

print("\nType 'exit' to quit.")


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
            "Please enter a query."
        )

        continue


    # ========================================================
    # EXTRACT FILTERS
    # ========================================================

    filters = extract_filters(
        query
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n===== EXTRACTED FILTERS ====="
    )

    for key, value in filters.items():

        print(
            f"{key:12} → {value}"
        )