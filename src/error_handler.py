import logging


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("jewellery_rag")


# ============================================================
# USER-FRIENDLY ERROR
# ============================================================

def user_friendly_error(error):

    error_text = str(error).lower()

    if "timeout" in error_text:

        return (
            "The service took too long to respond. "
            "Please try again."
        )

    if "rate limit" in error_text or "429" in error_text:

        return (
            "The service is currently busy. "
            "Please try again in a moment."
        )

    if "connection" in error_text:

        return (
            "I couldn't connect to the AI service. "
            "Please try again."
        )

    if "api" in error_text:

        return (
            "The AI service is temporarily unavailable. "
            "Please try again."
        )

    return (
        "Something went wrong while processing "
        "your request. Please try again."
    )


# ============================================================
# LOG ERROR
# ============================================================

def log_error(
    error,
    context=""
):

    logger.exception(
        "Application error | context=%s | error=%s",
        context,
        error
    )