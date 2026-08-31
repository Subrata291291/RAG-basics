from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME
)

from src.prompt_loader import load_prompts
from src.retry import retry_call


# ============================================================
# OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)


# ============================================================
# LOAD PROMPTS
# ============================================================

PROMPTS = load_prompts()


# ============================================================
# LOW-LEVEL LLM CALL
# ============================================================

def call_llm(messages):
    """
    Send a request to OpenRouter.

    Retry is handled centrally by retry_call().

    This function also checks for an empty OpenRouter
    response and provides useful diagnostic information.
    """

    def request():

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            timeout=30
        )

        # ----------------------------------------------------
        # CHECK FOR EMPTY RESPONSE
        # ----------------------------------------------------

        if not response.choices:

            error_details = []

            # OpenRouter may provide an error object
            if hasattr(response, "error") and response.error:

                error_details.append(
                    f"error={response.error}"
                )

            # Include model information if available
            if hasattr(response, "model") and response.model:

                error_details.append(
                    f"model={response.model}"
                )

            # Include response ID if available
            if hasattr(response, "id") and response.id:

                error_details.append(
                    f"id={response.id}"
                )

            # If nothing useful was available
            if not error_details:

                error_details.append(
                    f"response={response}"
                )

            details = " | ".join(error_details)

            raise RuntimeError(
                "OpenRouter returned no choices. "
                f"Details: {details}"
            )

        return response

    # --------------------------------------------------------
    # RETRY
    # --------------------------------------------------------

    return retry_call(
        request,
        max_retries=3,
        base_delay=1
    )


# ============================================================
# GENERIC ANSWER GENERATOR
# ============================================================

def generate_with_prompt(
    system_prompt,
    user_prompt
):
    """
    Generic LLM generation function.

    All LLM requests pass through call_llm(),
    which provides retry and timeout handling.
    """

    response = call_llm(
        [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    # --------------------------------------------------------
    # SAFELY EXTRACT RESPONSE
    # --------------------------------------------------------

    if not response.choices:

        raise RuntimeError(
            "OpenRouter returned no choices."
        )

    message = response.choices[0].message

    if message is None:

        raise RuntimeError(
            "OpenRouter returned an empty message."
        )

    answer = message.content

    if not answer:

        raise RuntimeError(
            "OpenRouter returned empty content."
        )

    return answer.strip()


# ============================================================
# PRODUCT ANSWER
# ============================================================

def generate_product_answer(
    query,
    context
):
    """
    Generate an answer about jewellery products.

    The answer is based only on the retrieved product
    context supplied by the RAG pipeline.
    """

    prompt = PROMPTS["product"].format(
        query=query,
        context=context
    )

    return generate_with_prompt(
        PROMPTS["system"],
        prompt
    )


# ============================================================
# KNOWLEDGE ANSWER
# ============================================================

def generate_knowledge_answer(
    query,
    context
):
    """
    Generate an answer using store knowledge documents.

    Examples:
        - Shipping policy
        - Return policy
        - Privacy policy
        - Terms and conditions
    """

    prompt = PROMPTS["knowledge"].format(
        query=query,
        context=context
    )

    return generate_with_prompt(
        PROMPTS["system"],
        prompt
    )


# ============================================================
# FOLLOW-UP ANSWER
# ============================================================

def generate_followup_answer(
    query,
    context,
    history
):
    """
    Generate an answer to a follow-up question.

    Uses:
        - Current retrieved product context
        - Conversation history
        - Current user question
    """

    prompt = PROMPTS["followup"].format(
        query=query,
        context=context,
        history=history
    )

    return generate_with_prompt(
        PROMPTS["system"],
        prompt
    )


# ============================================================
# NORMAL CONVERSATION ANSWER
# ============================================================

def generate_normal_answer(
    query,
    history
):
    """
    Generate a response for normal conversation.

    Examples:

        Hello
        Hi
        Thanks
        What can you do?

    Normal conversation should NOT require product
    retrieval or knowledge-document retrieval.
    """

    prompt = PROMPTS["normal"].format(
        query=query,
        history=history
    )

    return generate_with_prompt(
        PROMPTS["system"],
        prompt
    )