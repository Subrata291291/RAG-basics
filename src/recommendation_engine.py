# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

from typing import List, Dict, Optional


# ============================================================
# BASIC HELPERS
# ============================================================

def _price(product):
    """
    Safely get product price.
    """

    try:
        return float(product["price"])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _score(result):
    """
    Safely get semantic score.
    """

    try:
        return float(result.get("score", 0))
    except (TypeError, ValueError):
        return 0.0


def _product(result):
    """
    Safely get product object from search result.
    """

    return result.get("product", {})


# ============================================================
# CHEAPEST PRODUCT
# ============================================================

def cheapest_product(results):
    """
    Return the cheapest product from the supplied results.

    IMPORTANT:
    This does NOT search the catalog.
    It only considers the supplied results.
    """

    if not results:
        return None

    valid_results = [
        result
        for result in results
        if isinstance(result, dict)
        and isinstance(result.get("product"), dict)
        and "price" in result.get("product", {})
    ]

    if not valid_results:
        return None

    return min(
        valid_results,
        key=lambda result: _price(
            _product(result)
        )
    )


# ============================================================
# BEST SEMANTIC MATCH
# ============================================================

def best_match(results):
    """
    Return the highest semantic-score product.
    """

    if not results:
        return None

    valid_results = [
        result
        for result in results
        if isinstance(result, dict)
        and isinstance(result.get("product"), dict)
    ]

    if not valid_results:
        return None

    return max(
        valid_results,
        key=_score
    )


# ============================================================
# SORT BY PRICE
# ============================================================

def sort_by_price(
    results,
    ascending=True
):
    """
    Sort supplied products by price.
    """

    return sorted(
        results,
        key=lambda result: _price(
            _product(result)
        ),
        reverse=not ascending
    )


# ============================================================
# SORT BY RELEVANCE
# ============================================================

def sort_by_relevance(results):
    """
    Sort products by semantic similarity.
    """

    return sorted(
        results,
        key=_score,
        reverse=True
    )


# ============================================================
# BUDGET FILTER
# ============================================================

def within_budget(
    results,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None
):
    """
    Filter supplied results by price.

    This is an additional safety layer.
    """

    filtered = []

    for result in results:

        product = _product(result)

        if not product:
            continue

        price = _price(product)

        if price == float("inf"):
            continue

        if (
            max_price is not None
            and price > float(max_price)
        ):
            continue

        if (
            min_price is not None
            and price < float(min_price)
        ):
            continue

        filtered.append(result)

    return filtered


# ============================================================
# RECOMMEND TOP PRODUCTS
# ============================================================

def recommend(
    results,
    max_results=5,
    sort_by="relevance"
):
    """
    Return recommended products from the retrieved result set.

    sort_by:
        relevance
        price_low
        price_high
    """

    if not results:
        return []

    if sort_by == "price_low":

        ordered = sort_by_price(
            results,
            ascending=True
        )

    elif sort_by == "price_high":

        ordered = sort_by_price(
            results,
            ascending=False
        )

    else:

        ordered = sort_by_relevance(
            results
        )

    return ordered[:max_results]


# ============================================================
# RECOMMENDATION SUMMARY
# ============================================================

def build_recommendation_summary(results):
    """
    Build deterministic recommendation information.

    This information can then be passed to the LLM.
    """

    if not results:
        return {
            "count": 0,
            "cheapest": None,
            "best_match": None,
            "price_min": None,
            "price_max": None
        }

    products = [
        _product(result)
        for result in results
        if _product(result)
    ]

    if not products:
        return {
            "count": 0,
            "cheapest": None,
            "best_match": None,
            "price_min": None,
            "price_max": None
        }

    cheapest = cheapest_product(results)
    best = best_match(results)

    prices = [
        _price(product)
        for product in products
        if _price(product) != float("inf")
    ]

    return {
        "count": len(products),

        "cheapest": (
            cheapest["product"]
            if cheapest
            else None
        ),

        "best_match": (
            best["product"]
            if best
            else None
        ),

        "price_min": (
            min(prices)
            if prices
            else None
        ),

        "price_max": (
            max(prices)
            if prices
            else None
        )
    }



    # ============================================================
# RECOMMENDATION INTENT
# ============================================================

def detect_recommendation_intent(query):
    """
    Detect what type of recommendation the user wants.

    Returns:
        cheapest
        best
        price_low
        price_high
        None
    """

    if not query:
        return None

    text = query.lower().strip()

    # --------------------------------------------------------
    # CHEAPEST
    # --------------------------------------------------------

    cheapest_patterns = [
        "cheapest",
        "cheapest one",
        "cheapest product",
        "least expensive",
        "lowest price",
        "lowest priced",
        "most affordable",
        "budget friendly",
        "budget-friendly",
        "inexpensive",
        "which one costs less",
        "which costs less"
    ]

    for pattern in cheapest_patterns:

        if pattern in text:
            return "cheapest"

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    best_patterns = [
        "best",
        "best one",
        "best product",
        "best option",
        "which is best",
        "which one is best",
        "recommend one",
        "recommend me",
        "recommend",
        "top choice",
        "top option",
        "ideal one"
    ]

    for pattern in best_patterns:

        if pattern in text:
            return "best"

    # --------------------------------------------------------
    # LOWEST PRICE
    # --------------------------------------------------------

    if (
        "lowest price" in text
        or "low to high" in text
        or "cheapest first" in text
    ):
        return "price_low"

    # --------------------------------------------------------
    # HIGHEST PRICE
    # --------------------------------------------------------

    if (
        "highest price" in text
        or "high to low" in text
        or "most expensive" in text
    ):
        return "price_high"

    return None


# ============================================================
# APPLY RECOMMENDATION
# ============================================================

def get_recommendation(
    results,
    intent
):
    """
    Apply a deterministic recommendation to retrieved results.

    Returns:
        recommendation result or None
    """

    if not results or not intent:
        return None

    if intent == "cheapest":

        return cheapest_product(results)

    if intent == "best":

        return best_match(results)

    if intent == "price_low":

        ordered = sort_by_price(
            results,
            ascending=True
        )

        return ordered[0] if ordered else None

    if intent == "price_high":

        ordered = sort_by_price(
            results,
            ascending=False
        )

        return ordered[0] if ordered else None

    return None


# ============================================================
# RECOMMENDATION CONTEXT
# ============================================================

def recommendation_context(
    results,
    intent
):
    """
    Convert deterministic recommendation information
    into safe context for the LLM.

    The LLM is told the result calculated by Python,
    rather than being asked to calculate it itself.
    """

    recommendation = get_recommendation(
        results,
        intent
    )

    if not recommendation:
        return ""

    product = recommendation["product"]

    if intent == "cheapest":

        return f"""
RECOMMENDATION RESULT:

Recommendation type: cheapest

The cheapest product among the retrieved products is:

ID: {product['id']}
Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}
Price: ₹{product['price']}
Description: {product['description']}

IMPORTANT:
This recommendation was calculated programmatically from
the retrieved product results. Do not change the product,
price, ID, karat, or other attributes.
"""

    if intent == "best":

        return f"""
RECOMMENDATION RESULT:

Recommendation type: best match

The highest semantic-match product among the retrieved
products is:

ID: {product['id']}
Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}
Price: ₹{product['price']}
Description: {product['description']}
Semantic Score: {recommendation.get('score', 0):.3f}

IMPORTANT:
This recommendation was calculated programmatically.
Do not invent or change any product information.
"""

    return f"""
RECOMMENDATION RESULT:

Recommendation type: {intent}

Recommended product:

ID: {product['id']}
Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}
Price: ₹{product['price']}
Description: {product['description']}

IMPORTANT:
Use only the supplied product information.
"""