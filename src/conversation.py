class ConversationState:

    def __init__(self):

        self.history = []

        self.last_products = []


    # ========================================================
    # ADD MESSAGE
    # ========================================================

    def add_message(
        self,
        user,
        assistant
    ):

        self.history.append(
            {
                "user": user,
                "assistant": assistant
            }
        )

        # Keep only recent history.
        self.history = self.history[-6:]


    # ========================================================
    # SAVE PRODUCTS
    # ========================================================

    def set_products(
        self,
        products
    ):

        self.last_products = products


    # ========================================================
    # HISTORY TEXT
    # ========================================================

    def history_text(self):

        if not self.history:

            return "No previous conversation."

        lines = []

        for item in self.history:

            lines.append(
                f"Customer: {item['user']}"
            )

            lines.append(
                f"Assistant: {item['assistant']}"
            )

        return "\n".join(lines)


    # ========================================================
    # PRODUCT CONTEXT
    # ========================================================

    def product_context(self):

        if not self.last_products:

            return "No previous products."

        lines = []

        for rank, result in enumerate(
            self.last_products,
            start=1
        ):

            product = result["product"]

            lines.append(
                f"""
Product {rank}:
ID: {product['id']}
Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}
Price: ₹{product['price']}
Description: {product['description']}
"""
            )

        return "\n".join(lines)