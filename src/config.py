from pathlib import Path
import os
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"

PRODUCTS_FILE = DATA_DIR / "products.json"

KNOWLEDGE_DIR = DATA_DIR / "knowledge"


# ============================================================
# PROMPT DIRECTORY
# ============================================================

PROMPTS_DIR = BASE_DIR / "prompts"


# ============================================================
# API CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
)


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "openrouter/free"
)


# ============================================================
# RAG CONFIGURATION
# ============================================================

TOP_K_PRODUCTS = int(
    os.getenv("TOP_K_PRODUCTS", "5")
)

TOP_K_KNOWLEDGE = int(
    os.getenv("TOP_K_KNOWLEDGE", "3")
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "all-MiniLM-L6-v2"
)


# ============================================================
# VALIDATION
# ============================================================

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from .env"
    )