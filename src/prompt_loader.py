from src.config import PROMPTS_DIR


# ============================================================
# LOAD ONE PROMPT
# ============================================================

def load_prompt(filename):

    file_path = PROMPTS_DIR / filename

    if not file_path.exists():

        raise FileNotFoundError(
            f"Prompt file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read().strip()


# ============================================================
# LOAD ALL APPLICATION PROMPTS
# ============================================================

def load_prompts():

    return {

        "system": load_prompt(
            "system_prompt.txt"
        ),

        "product": load_prompt(
            "product_prompt.txt"
        ),

        "knowledge": load_prompt(
            "knowledge_prompt.txt"
        ),

        "followup": load_prompt(
            "followup_prompt.txt"
        ),

        "normal": load_prompt(
            "normal_chat_prompt.txt"
        ),

        "validation": load_prompt(
            "validation_prompt.txt"
        )
    }