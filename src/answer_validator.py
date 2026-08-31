# src/answer_validator.py

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
# CALL VALIDATOR LLM
# ============================================================

def call_validator_llm(
    system_prompt,
    user_prompt
):

    def request():

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0,

            timeout=30
        )


        # ----------------------------------------------------
        # RESPONSE SAFETY CHECK
        # ----------------------------------------------------

        if response is None:

            raise RuntimeError(
                "Validator returned no response."
            )


        if not response.choices:

            raise RuntimeError(
                "Validator returned no choices."
            )


        choice = response.choices[0]


        if choice is None:

            raise RuntimeError(
                "Validator returned an empty choice."
            )


        message = choice.message


        if message is None:

            raise RuntimeError(
                "Validator returned an empty message."
            )


        content = message.content


        if not content:

            raise RuntimeError(
                "Validator returned empty content."
            )


        return content.strip()


    return retry_call(

        request,

        max_retries=3,

        base_delay=1

    )


# ============================================================
# PARSE VALIDATION RESULT
# ============================================================

def parse_validation_result(
    response
):

    if response is None:

        return None


    text = str(
        response
    ).strip().lower()


    # --------------------------------------------------------
    # VALID
    # --------------------------------------------------------

    if text.startswith("valid"):

        return True


    # --------------------------------------------------------
    # INVALID
    # --------------------------------------------------------

    if text.startswith("invalid"):

        return False


    # --------------------------------------------------------
    # SEARCH FOR EXPLICIT WORDS
    # --------------------------------------------------------

    if "valid: yes" in text:

        return True


    if "valid: no" in text:

        return False


    if "validation: passed" in text:

        return True


    if "validation: failed" in text:

        return False


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return None


# ============================================================
# VALIDATE ANSWER
# ============================================================

def validate_answer(
    query,
    context,
    answer
):

    # ========================================================
    # INPUT SAFETY
    # ========================================================

    if query is None:

        return (
            False,
            "Validation failed: query is empty."
        )


    if context is None:

        return (
            False,
            "Validation failed: retrieved context is empty."
        )


    if answer is None:

        return (
            False,
            "Validation failed: generated answer is empty."
        )


    query = str(
        query
    ).strip()


    context = str(
        context
    ).strip()


    answer = str(
        answer
    ).strip()


    if not query:

        return (
            False,
            "Validation failed: query is empty."
        )


    if not context:

        return (
            False,
            "Validation failed: retrieved context is empty."
        )


    if not answer:

        return (
            False,
            "Validation failed: generated answer is empty."
        )


    # ========================================================
    # BUILD VALIDATION PROMPT
    # ========================================================

    try:

        prompt = PROMPTS["validation"].format(

            query=query,

            context=context,

            answer=answer

        )

    except KeyError:

        # ----------------------------------------------------
        # FALLBACK PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are an answer validation system.

Determine whether the generated answer is supported
by the provided store information.

USER QUESTION:
{query}

STORE INFORMATION:
{context}

GENERATED ANSWER:
{answer}

Return exactly one of:

VALID
INVALID

Return VALID only if the answer is supported by
the store information.
"""


    # ========================================================
    # CALL VALIDATOR
    # ========================================================

    try:

        validation_response = (
            call_validator_llm(

                PROMPTS["system"],

                prompt

            )
        )


    except Exception as e:

        return (
            False,
            f"Validation service error: {str(e)}"
        )


    # ========================================================
    # EMPTY VALIDATOR RESPONSE
    # ========================================================

    if validation_response is None:

        return (
            False,
            "Validator returned no response."
        )


    validation_response = str(
        validation_response
    ).strip()


    if not validation_response:

        return (
            False,
            "Validator returned an empty response."
        )


    # ========================================================
    # PARSE RESULT
    # ========================================================

    result = parse_validation_result(
        validation_response
    )


    # ========================================================
    # UNKNOWN RESULT
    # ========================================================

    if result is None:

        return (
            False,
            "Validator returned an unclear result: "
            f"{validation_response}"
        )


    # ========================================================
    # VALID
    # ========================================================

    if result:

        return (
            True,
            "Answer is supported by the retrieved context."
        )


    # ========================================================
    # INVALID
    # ========================================================

    return (
        False,
        "Generated answer contains information "
        "that could not be verified from the retrieved context."
    )# ============================================================
# ANSWER VALIDATOR
# ============================================================

import re

from src.answer_generator import generate_with_prompt
from src.prompt_loader import load_prompts


# ============================================================
# LOAD PROMPTS
# ============================================================

PROMPTS = load_prompts()


# ============================================================
# VALIDATOR RESPONSE PARSER
# ============================================================

def _parse_validation_response(response):
    """
    Convert the validator LLM response into:

        (True, reason)
    or
        (False, reason)

    The validator is intentionally strict.
    """

    if not response:
        return False, "Validator returned an empty response."

    text = response.strip()

    # --------------------------------------------------------
    # VALID
    # --------------------------------------------------------

    if re.fullmatch(
        r"VALID\s*",
        text,
        flags=re.IGNORECASE
    ):
        return True, "Answer is supported by the retrieved context."

    # --------------------------------------------------------
    # INVALID
    # --------------------------------------------------------

    invalid_match = re.match(
        r"^INVALID\s*(?::|-)?\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if invalid_match:

        reason = invalid_match.group(1).strip()

        if not reason:
            reason = "Answer contains unsupported information."

        return False, reason

    # --------------------------------------------------------
    # FAIL CLOSED
    # --------------------------------------------------------

    return False, (
        "Validator returned an unrecognized response."
    )


# ============================================================
# VALIDATE ANSWER
# ============================================================

def validate_answer(
    query,
    context,
    answer
):
    """
    Validate a generated RAG answer against retrieved context.

    IMPORTANT:
    Validation fails closed. If the validator itself returns an
    unexpected response or raises an exception, the answer is
    considered unverified.
    """

    if not query or not answer:
        return False, "Question or answer is empty."

    if not context:
        return False, "No retrieved context was supplied."

    try:

        validation_prompt = PROMPTS["validation"].format(
            query=query,
            context=context,
            answer=answer
        )

    except KeyError:

        return False, (
            "Validation prompt is not configured."
        )

    except Exception as e:

        return False, (
            f"Could not build validation prompt: {e}"
        )

    try:

        response = generate_with_prompt(
            PROMPTS["system"],
            validation_prompt
        )

    except Exception as e:

        return False, (
            f"Validator request failed: {e}"
        )

    return _parse_validation_response(
        response
    )
