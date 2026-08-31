import time
import random


# ============================================================
# RETRY CONFIGURATION
# ============================================================

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 10.0


# ============================================================
# CHECK WHETHER ERROR IS RETRYABLE
# ============================================================

def is_retryable_error(error):
    """
    Decide whether an exception is likely to be temporary.

    We retry:
        - timeout errors
        - connection errors
        - HTTP 429 rate limits
        - HTTP 500/502/503/504 server errors
        - OpenRouter temporary/aborted operations

    We do NOT retry obvious permanent errors such as:
        - invalid API key
        - invalid request
        - authentication errors
        - malformed configuration
    """

    error_text = str(error).lower()


    # --------------------------------------------------------
    # Python/network timeout errors
    # --------------------------------------------------------

    if isinstance(
        error,
        (
            TimeoutError,
            ConnectionError
        )
    ):

        return True


    # --------------------------------------------------------
    # Common transient HTTP errors
    # --------------------------------------------------------

    transient_codes = (
        "429",
        "500",
        "502",
        "503",
        "504"
    )


    for code in transient_codes:

        if code in error_text:

            return True


    # --------------------------------------------------------
    # OpenRouter / provider temporary errors
    # --------------------------------------------------------

    transient_messages = [

        "timeout",

        "timed out",

        "temporarily unavailable",

        "temporary failure",

        "connection reset",

        "connection aborted",

        "connection error",

        "server error",

        "service unavailable",

        "operation was aborted",

        "gateway timeout",

        "bad gateway",

        "rate limit",

        "too many requests",

        "overloaded",

        "no choices"

    ]


    for message in transient_messages:

        if message in error_text:

            return True


    # --------------------------------------------------------
    # Everything else is treated as permanent
    # --------------------------------------------------------

    return False


# ============================================================
# RETRY CALL
# ============================================================

def retry_call(
    function,
    max_retries=DEFAULT_MAX_RETRIES,
    base_delay=DEFAULT_BASE_DELAY,
    max_delay=DEFAULT_MAX_DELAY
):
    """
    Execute a function with retry + exponential backoff.

    Example:

        result = retry_call(
            request,
            max_retries=3,
            base_delay=1
        )

    IMPORTANT:

    max_retries means TOTAL ATTEMPTS.

    Therefore:

        max_retries=3

    means:

        Attempt 1
        Attempt 2
        Attempt 3

    It does NOT mean 1 initial attempt + 3 retries.
    """


    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    if not callable(function):

        raise TypeError(
            "retry_call() expected a callable function."
        )


    if max_retries < 1:

        raise ValueError(
            "max_retries must be at least 1."
        )


    if base_delay < 0:

        raise ValueError(
            "base_delay cannot be negative."
        )


    if max_delay < base_delay:

        raise ValueError(
            "max_delay must be greater than or equal "
            "to base_delay."
        )


    last_error = None


    # ========================================================
    # ATTEMPTS
    # ========================================================

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            # ------------------------------------------------
            # EXECUTE FUNCTION
            # ------------------------------------------------

            return function()


        except Exception as error:

            last_error = error


            # ------------------------------------------------
            # CHECK RETRYABILITY
            # ------------------------------------------------

            retryable = is_retryable_error(
                error
            )


            # ------------------------------------------------
            # PERMANENT ERROR
            # ------------------------------------------------

            if not retryable:

                print()

                print(
                    "LLM request failed with "
                    "a non-retryable error:"
                )

                print(
                    str(error)
                )

                raise


            # ------------------------------------------------
            # LAST ATTEMPT
            # ------------------------------------------------

            if attempt >= max_retries:

                print()

                print(
                    f"LLM request failed after "
                    f"{attempt}/{max_retries} attempts."
                )

                print(
                    f"Error: {error}"
                )

                raise


            # ------------------------------------------------
            # EXPONENTIAL BACKOFF
            # ------------------------------------------------

            delay = min(

                base_delay * (
                    2 ** (attempt - 1)
                ),

                max_delay

            )


            # ------------------------------------------------
            # JITTER
            # ------------------------------------------------
            #
            # Adds a small random amount so multiple
            # clients do not retry at exactly the same time.
            #

            jitter = random.uniform(
                0,
                0.25
            )


            total_delay = (
                delay + jitter
            )


            # ------------------------------------------------
            # LOG RETRY
            # ------------------------------------------------

            next_attempt = (
                attempt + 1
            )


            print()

            print(
                f"LLM request failed. "
                f"Retrying attempt "
                f"{next_attempt}/{max_retries} "
                f"after {total_delay:.2f} seconds."
            )

            print(
                f"Error: {error}"
            )


            # ------------------------------------------------
            # WAIT
            # ------------------------------------------------

            time.sleep(
                total_delay
            )


    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    if last_error is not None:

        raise last_error


    raise RuntimeError(
        "retry_call() failed without an exception."
    )