from src.data_loader import (
    load_products,
    load_knowledge_documents
)

from src.product_search import (
    build_product_embeddings,
    filtered_semantic_search
)

from src.query_understanding import (
    understand_query
)

from src.knowledge_search import (
    build_knowledge_embeddings,
    search_knowledge
)

from src.query_router import (
    classify_query
)

from src.conversation import (
    ConversationState
)

from src.answer_generator import (
    generate_product_answer,
    generate_knowledge_answer,
    generate_followup_answer,
    generate_normal_answer
)

from src.answer_validator import (
    validate_answer
)

from src.config import (
    TOP_K_PRODUCTS,
    TOP_K_KNOWLEDGE
)


# ============================================================
# CHATBOT
# ============================================================

class JewelleryChatbot:

    def __init__(self):

        print()
        print(
            "Loading jewellery chatbot..."
        )

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        self.products = load_products()

        self.knowledge = (
            load_knowledge_documents()
        )

        print(
            f"Loaded {len(self.products)} products."
        )

        print(
            f"Loaded {len(self.knowledge)} knowledge documents."
        )


        # ----------------------------------------------------
        # PRODUCT EMBEDDINGS
        # ----------------------------------------------------

        print(
            "Creating product embeddings..."
        )

        self.product_embeddings = (
            build_product_embeddings(
                self.products
            )
        )

        print(
            "Product embeddings created."
        )


        # ----------------------------------------------------
        # KNOWLEDGE EMBEDDINGS
        # ----------------------------------------------------

        if self.knowledge:

            print(
                "Creating knowledge embeddings..."
            )

            self.knowledge_embeddings = (
                build_knowledge_embeddings(
                    self.knowledge
                )
            )

            print(
                "Knowledge embeddings created."
            )

        else:

            self.knowledge_embeddings = []


        # ----------------------------------------------------
        # CONVERSATION
        # ----------------------------------------------------

        self.conversation = (
            ConversationState()
        )


        print(
            "Chatbot ready."
        )


    # ========================================================
    # PRODUCT CONTEXT
    # ========================================================

    def product_context(
        self,
        results
    ):

        lines = []

        for rank, result in enumerate(
            results,
            start=1
        ):

            product = result["product"]

            material_type = product.get(
                "material_type"
            )

            material_line = ""

            if material_type:

                material_line = (
                    f"Material Type: "
                    f"{material_type}\n"
                )


            lines.append(
                f"""
Product {rank}:
ID: {product.get('id', '')}
Name: {product.get('name', '')}
Category: {product.get('category', '')}
Metal: {product.get('metal', '')}
{material_line}Karat: {product.get('karat', '')}
Price: ₹{product.get('price', '')}
Description: {product.get('description', '')}
Semantic Score: {result.get('score', 0):.3f}
"""
            )

        return "\n".join(lines)


    # ========================================================
    # KNOWLEDGE CONTEXT
    # ========================================================

    def knowledge_context(
        self,
        results
    ):

        lines = []

        for rank, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]

            lines.append(
                f"""
Document {rank}:
File: {document.get('document', '')}
Score: {result.get('score', 0):.3f}

Content:
{document.get('content', '')}
"""
            )

        return "\n".join(lines)


    # ========================================================
    # PRINT QUERY FILTERS
    # ========================================================

    def print_query_filters(
        self,
        filters
    ):

        print()

        print(
            "===== QUERY UNDERSTANDING ====="
        )

        print(
            f"Original query: "
            f"{filters.get('original_query')}"
        )

        print(
            f"Corrected query: "
            f"{filters.get('corrected_query')}"
        )

        print(
            f"Category: "
            f"{filters.get('category')}"
        )

        print(
            f"Metal: "
            f"{filters.get('metal')}"
        )

        print(
            f"Material type: "
            f"{filters.get('material_type')}"
        )

        print(
            f"Karat: "
            f"{filters.get('karat')}"
        )

        print(
            f"Min price: "
            f"{filters.get('min_price')}"
        )

        print(
            f"Max price: "
            f"{filters.get('max_price')}"
        )

        print(
            f"Sort: "
            f"{filters.get('sort_by')} "
            f"{filters.get('sort_order')}"
        )


    # ========================================================
    # HANDLE QUERY
    # ========================================================

    def ask(
        self,
        query
    ):

        query = query.strip()


        # ----------------------------------------------------
        # EMPTY QUERY
        # ----------------------------------------------------

        if not query:

            return (
                "Please enter a question."
            )


        # ----------------------------------------------------
        # CLASSIFY QUERY
        # ----------------------------------------------------

        try:

            query_type = classify_query(

                query,

                has_previous_products=bool(
                    self.conversation.last_products
                )

            )

        except Exception as e:

            print()

            print(
                "QUERY ROUTER ERROR:"
            )

            print(
                str(e)
            )


            return (
                "I'm sorry, I couldn't understand "
                "your request right now. Please try again."
            )


        print()

        print(
            f"Query type: {query_type}"
        )


        # ====================================================
        # NORMAL CONVERSATION
        # ====================================================

        if query_type == "normal":

            print()

            print(
                "===== NORMAL CONVERSATION ====="
            )


            history = (
                self.conversation.history_text()
            )


            try:

                answer = generate_normal_answer(

                    query=query,

                    history=history

                )


            except Exception as e:

                print()

                print(
                    "NORMAL CHAT ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I'm having trouble "
                    "responding right now. Please try again."
                )


            # ------------------------------------------------
            # EMPTY ANSWER
            # ------------------------------------------------

            if not answer:

                return (
                    "I'm sorry, I couldn't generate "
                    "a response right now. Please try again."
                )


            # ------------------------------------------------
            # NORMAL CHAT DOES NOT NEED RAG VALIDATION
            # ------------------------------------------------

            print()

            print(
                "Normal conversation - "
                "validation skipped."
            )


            self.conversation.add_message(

                query,

                answer

            )


            return answer


        # ====================================================
        # FOLLOW-UP
        # ====================================================

        if query_type == "followup":

            print()

            print(
                "===== FOLLOW-UP ====="
            )


            context = (
                self.conversation.product_context()
            )


            history = (
                self.conversation.history_text()
            )


            # ------------------------------------------------
            # NO PREVIOUS PRODUCTS
            # ------------------------------------------------

            if not context:

                return (
                    "I don't have a previous product "
                    "selection to refer to. Please tell "
                    "me which jewellery you're interested in."
                )


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            try:

                answer = generate_followup_answer(

                    query,

                    context,

                    history

                )


            except Exception as e:

                print()

                print(
                    "FOLLOW-UP LLM ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't process that "
                    "follow-up question right now. "
                    "Please try again."
                )


            # ------------------------------------------------
            # EMPTY ANSWER
            # ------------------------------------------------

            if not answer:

                print()

                print(
                    "FOLLOW-UP ERROR: "
                    "LLM returned an empty answer."
                )


                return (
                    "I'm sorry, I couldn't generate an "
                    "answer right now. Please try again."
                )


            validation_context = context


        # ====================================================
        # PRODUCT
        # ====================================================

        elif query_type == "product":

            print()

            print(
                "===== PRODUCT SEARCH ====="
            )


            # =================================================
            # QUERY UNDERSTANDING
            # =================================================

            try:

                filters = understand_query(
                    query
                )

            except Exception as e:

                print()

                print(
                    "QUERY UNDERSTANDING ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't understand "
                    "your product requirements right now. "
                    "Please try again."
                )


            # ------------------------------------------------
            # DISPLAY UNDERSTANDING
            # ------------------------------------------------

            self.print_query_filters(
                filters
            )


            # =================================================
            # FILTERED + SEMANTIC SEARCH
            # =================================================

            try:

                results = filtered_semantic_search(

                    query,

                    self.products,

                    self.product_embeddings,

                    filters,

                    TOP_K_PRODUCTS

                )


            except Exception as e:

                print()

                print(
                    "PRODUCT SEARCH ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't search the "
                    "product catalog right now."
                )


            # =================================================
            # NO RESULTS
            # =================================================

            if not results:

                print()

                print(
                    "No matching products found."
                )


                # ------------------------------------------------
                # SPECIAL MESSAGE FOR STRICT FILTERS
                # ------------------------------------------------

                material_type = filters.get(
                    "material_type"
                )

                category = filters.get(
                    "category"
                )

                metal = filters.get(
                    "metal"
                )

                max_price = filters.get(
                    "max_price"
                )


                if material_type:

                    readable_material = (
                        material_type.replace(
                            "_",
                            " "
                        )
                    )

                    return (
                        f"I couldn't find any "
                        f"{readable_material} products "
                        f"matching your requirements "
                        f"in the current catalog."
                    )


                if (
                    category
                    and metal
                    and max_price is not None
                ):

                    return (
                        f"I couldn't find any "
                        f"{metal} {category} products "
                        f"under ₹{max_price:,} "
                        f"in the current catalog."
                    )


                if (
                    category
                    and max_price is not None
                ):

                    return (
                        f"I couldn't find any "
                        f"{category} products under "
                        f"₹{max_price:,} "
                        f"in the current catalog."
                    )


                if (
                    metal
                    and max_price is not None
                ):

                    return (
                        f"I couldn't find any "
                        f"{metal} products under "
                        f"₹{max_price:,} "
                        f"in the current catalog."
                    )


                return (
                    "I couldn't find any products "
                    "matching your requirements "
                    "in the current catalog."
                )


            # =================================================
            # DISPLAY RESULTS
            # =================================================

            print()

            print(
                "===== MATCHING PRODUCTS ====="
            )


            for rank, result in enumerate(

                results,

                start=1

            ):

                product = result["product"]


                print(

                    f"{rank}. "

                    f"{product.get('name', '')} | "

                    f"₹{product.get('price', 0):,} | "

                    f"Score={result.get('score', 0):.3f}"

                )


            # =================================================
            # SORT RESULTS IF REQUESTED
            # =================================================

            sort_by = filters.get(
                "sort_by"
            )

            sort_order = filters.get(
                "sort_order"
            )


            if sort_by == "price":

                reverse = (
                    sort_order == "desc"
                )


                results = sorted(

                    results,

                    key=lambda item: float(
                        item["product"].get(
                            "price",
                            0
                        )
                    ),

                    reverse=reverse

                )


                print()

                print(
                    "Products sorted by price."
                )


            # =================================================
            # CONTEXT
            # =================================================

            context = (
                self.product_context(
                    results
                )
            )


            # =================================================
            # GENERATE ANSWER
            # =================================================

            try:

                answer = generate_product_answer(

                    query,

                    context

                )


            except Exception as e:

                print()

                print(
                    "PRODUCT LLM ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't generate "
                    "a product answer right now. "
                    "Please try again."
                )


            # ------------------------------------------------
            # EMPTY ANSWER
            # ------------------------------------------------

            if not answer:

                return (
                    "I'm sorry, I couldn't generate "
                    "an answer right now. Please try again."
                )


            validation_context = context


            # =================================================
            # SAVE PRODUCTS
            # =================================================

            self.conversation.set_products(
                results
            )


        # ====================================================
        # KNOWLEDGE
        # ====================================================

        elif query_type == "knowledge":

            print()

            print(
                "===== KNOWLEDGE SEARCH ====="
            )


            if not self.knowledge:

                return (
                    "The store knowledge documents "
                    "are currently unavailable."
                )


            # ------------------------------------------------
            # SEARCH
            # ------------------------------------------------

            try:

                results = search_knowledge(

                    query,

                    self.knowledge,

                    self.knowledge_embeddings,

                    TOP_K_KNOWLEDGE

                )


            except Exception as e:

                print()

                print(
                    "KNOWLEDGE SEARCH ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't search the "
                    "store information right now."
                )


            # ------------------------------------------------
            # NO RESULTS
            # ------------------------------------------------

            if not results:

                return (
                    "I couldn't find relevant information "
                    "in the store knowledge documents."
                )


            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            for rank, result in enumerate(

                results,

                start=1

            ):

                document = result["document"]


                print(

                    f"{rank}. "

                    f"{document.get('document', '')} | "

                    f"Score={result.get('score', 0):.3f}"

                )


            # ------------------------------------------------
            # CONTEXT
            # ------------------------------------------------

            context = (
                self.knowledge_context(
                    results
                )
            )


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            try:

                answer = generate_knowledge_answer(

                    query,

                    context

                )


            except Exception as e:

                print()

                print(
                    "KNOWLEDGE LLM ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't generate "
                    "an answer from the store information "
                    "right now. Please try again."
                )


            # ------------------------------------------------
            # EMPTY ANSWER
            # ------------------------------------------------

            if not answer:

                return (
                    "I'm sorry, I couldn't generate "
                    "an answer right now. Please try again."
                )


            validation_context = context


        # ====================================================
        # UNKNOWN TYPE
        # ====================================================

        else:

            return (
                "I'm not sure how to handle that request. "
                "Please ask about our jewellery, products, "
                "or store policies."
            )


        # ====================================================
        # ANSWER VALIDATION
        # ====================================================

        print()

        print(
            "===== ANSWER VALIDATION ====="
        )


        # ----------------------------------------------------
        # FINAL SAFETY CHECK
        # ----------------------------------------------------

        if not answer:

            print(
                "VALIDATION SKIPPED: "
                "No answer was generated."
            )


            return (
                "I couldn't generate an answer right now. "
                "Please try again."
            )


        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        try:

            validation_result = (
                validate_answer(

                    query,

                    validation_context,

                    answer

                )
            )


        except Exception as e:

            print()

            print(
                "Validation error:"
            )

            print(
                str(e)
            )


            return (
                "I generated an answer, but I couldn't "
                "verify it against the available store "
                "information. Please try again."
            )


        # ----------------------------------------------------
        # VALIDATION RESULT SAFETY CHECK
        # ----------------------------------------------------

        if (

            validation_result is None

            or not isinstance(
                validation_result,
                tuple
            )

            or len(validation_result) != 2

        ):

            print()

            print(
                "VALIDATION ERROR: "
                "Invalid validator response."
            )


            return (
                "I couldn't verify the answer against "
                "the available store information."
            )


        valid, validation_message = (
            validation_result
        )


        # ====================================================
        # VALIDATION PASSED
        # ====================================================

        if valid:

            print(
                "VALIDATION: PASSED"
            )

            final_answer = answer


        # ====================================================
        # VALIDATION FAILED
        # ====================================================

        else:

            print(
                "VALIDATION: FAILED"
            )

            print(
                validation_message
            )


            final_answer = (
                "I couldn't verify that answer "
                "from the available store information."
            )


        # ====================================================
        # MEMORY
        # ====================================================

        self.conversation.add_message(

            query,

            final_answer

        )


        # ====================================================
        # RETURN
        # ====================================================

        return final_answer