from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


# ============================================================
# EMBEDDING MODEL
# ============================================================

_model = None


def get_embedding_model():

    global _model

    if _model is None:

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_MODEL_NAME}"
        )

        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print("Embedding model loaded.")

    return _model


# ============================================================
# EMBED TEXT
# ============================================================

def embed_text(text):

    model = get_embedding_model()

    return model.encode(
        text,
        normalize_embeddings=True
    )


# ============================================================
# EMBED MANY TEXTS
# ============================================================

def embed_texts(texts):

    model = get_embedding_model()

    return model.encode(
        texts,
        normalize_embeddings=True
    )