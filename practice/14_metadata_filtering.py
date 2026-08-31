# ============================================================
# 14_metadata_filtering.py
# METADATA FILTERING FOR RAG
# ============================================================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PRODUCT DATA
# ============================================================

products = [

    {
        "id": "R001",
        "name": "Classic 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 22,
        "price": 18000,
        "in_stock": True
    },

    {
        "id": "R002",
        "name": "Elegant 22K Gold Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 22,
        "price": 25000,
        "in_stock": True
    },

    {
        "id": "R003",
        "name": "Diamond Ring",
        "category": "ring",
        "metal": "gold",
        "karat": 18,
        "price": 19500,
        "in_stock": True
    },

    {
        "id": "N001",
        "name": "22K Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": 22,
        "price": 35000,
        "in_stock": True
    },

    {
        "id": "N002",
        "name": "Diamond Gold Necklace",
        "category": "necklace",
        "metal": "gold",
        "karat": 18,
        "price": 45000,
        "in_stock": True
    },

    {
        "id": "B001",
        "name": "22K Gold Bracelet",
        "category": "bracelet",
        "metal": "gold",
        "karat": 22,
        "price": 22000,
        "in_stock": True
    },

    {
        "id": "E001",
        "name": "Gold Earrings",
        "category": "earrings",
        "metal": "gold",
        "karat": 22,
        "price": 15000,
        "in_stock": True
    },

    {
        "id": "R004",
        "name": "Silver Ring",
        "category": "ring",
        "metal": "silver",
        "karat": 0,
        "price": 8000,
        "in_stock": True
    }
]


# ============================================================
# 2. DISPLAY ALL PRODUCTS
# ============================================================

print("\n========================================")
print("        JEWELLERY PRODUCT DATABASE")
print("========================================")

for product in products:

    print(
        f"{product['id']} | "
        f"{product['name']} | "
        f"{product['category']} | "
        f"{product['metal']} | "
        f"{product['karat']}K | "
        f"₹{product['price']}"
    )


# ============================================================
# 3. METADATA FILTER FUNCTION
# ============================================================

def filter_products(
    products,
    category=None,
    metal=None,
    karat=None,
    max_price=None,
    min_price=None,
    in_stock=None
):

    filtered_products = []

    for product in products:

        # ----------------------------------------------------
        # CATEGORY FILTER
        # ----------------------------------------------------

        if category is not None:

            if product["category"] != category:
                continue


        # ----------------------------------------------------
        # METAL FILTER
        # ----------------------------------------------------

        if metal is not None:

            if product["metal"] != metal:
                continue


        # ----------------------------------------------------
        # KARAT FILTER
        # ----------------------------------------------------

        if karat is not None:

            if product["karat"] != karat:
                continue


        # ----------------------------------------------------
        # MAX PRICE FILTER
        # ----------------------------------------------------

        if max_price is not None:

            if product["price"] > max_price:
                continue


        # ----------------------------------------------------
        # MIN PRICE FILTER
        # ----------------------------------------------------

        if min_price is not None:

            if product["price"] < min_price:
                continue


        # ----------------------------------------------------
        # STOCK FILTER
        # ----------------------------------------------------

        if in_stock is not None:

            if product["in_stock"] != in_stock:
                continue


        # ----------------------------------------------------
        # PRODUCT PASSED ALL FILTERS
        # ----------------------------------------------------

        filtered_products.append(product)


    return filtered_products


# ============================================================
# 4. TEST FILTERING
# ============================================================

print("\n========================================")
print("        FILTER TEST")
print("========================================")


filtered = filter_products(
    products,
    category="ring",
    metal="gold",
    karat=22,
    max_price=20000
)


print("\n22K gold rings under ₹20,000:")


for product in filtered:

    print(
        f"\n{product['id']} - {product['name']}"
    )

    print(
        f"Price: ₹{product['price']}"
    )

    print(
        f"Stock: {product['in_stock']}"
    )


# ============================================================
# 5. ANOTHER TEST
# ============================================================

filtered = filter_products(
    products,
    category="ring",
    max_price=20000
)


print("\n========================================")
print("        RINGS UNDER ₹20,000")
print("========================================")


for product in filtered:

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
        f"Price: ₹{product['price']}"
    )


# ============================================================
# 6. GOLD PRODUCTS
# ============================================================

filtered = filter_products(
    products,
    metal="gold"
)


print("\n========================================")
print("        ALL GOLD PRODUCTS")
print("========================================")


for product in filtered:

    print(
        f"{product['name']} "
        f"- ₹{product['price']}"
    )