from src.recommendation_engine import (
    cheapest_product,
    best_match,
    sort_by_price,
    build_recommendation_summary
)


results = [

    {
        "product": {
            "id": "R001",
            "name": "Classic 22K Gold Ring",
            "price": 18000
        },
        "score": 0.436
    },

    {
        "product": {
            "id": "R005",
            "name": "Modern 22K Gold Ring",
            "price": 24000
        },
        "score": 0.429
    },

    {
        "product": {
            "id": "R003",
            "name": "Traditional 22K Gold Ring",
            "price": 27000
        },
        "score": 0.414
    },

    {
        "product": {
            "id": "R002",
            "name": "Elegant 22K Gold Ring",
            "price": 25000
        },
        "score": 0.402
    }
]


print()
print("===== CHEAPEST =====")

result = cheapest_product(results)

print(
    result["product"]["name"],
    "₹",
    result["product"]["price"]
)


print()
print("===== BEST MATCH =====")

result = best_match(results)

print(
    result["product"]["name"],
    "Score =",
    result["score"]
)


print()
print("===== PRICE SORT =====")

for result in sort_by_price(results):

    product = result["product"]

    print(
        product["name"],
        "₹",
        product["price"]
    )


print()
print("===== SUMMARY =====")

summary = build_recommendation_summary(
    results
)

print(summary)