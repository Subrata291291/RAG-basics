from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 2. CREATE THREE SENTENCES
# ============================================================

sentence_a = "The iPhone 15 has a powerful camera."

sentence_b = "The iPhone 15 has a great camera."

sentence_c = "The iPhone 15 uses the A16 Bionic chip."


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings...")

embedding_a = model.encode(sentence_a)

embedding_b = model.encode(sentence_b)

embedding_c = model.encode(sentence_c)

print("Embeddings created.")


# ============================================================
# 4. DISPLAY EMBEDDING INFORMATION
# ============================================================

print("\n===== EMBEDDING INFORMATION =====")

print(
    "Sentence A embedding shape:",
    embedding_a.shape
)

print(
    "Sentence B embedding shape:",
    embedding_b.shape
)

print(
    "Sentence C embedding shape:",
    embedding_c.shape
)


# ============================================================
# 5. CALCULATE COSINE SIMILARITY
# ============================================================

similarity_ab = cosine_similarity(
    [embedding_a],
    [embedding_b]
)[0][0]


similarity_ac = cosine_similarity(
    [embedding_a],
    [embedding_c]
)[0][0]


similarity_bc = cosine_similarity(
    [embedding_b],
    [embedding_c]
)[0][0]


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n===== COSINE SIMILARITY =====")

print(
    "A vs B:",
    round(similarity_ab, 3)
)

print(
    "A vs C:",
    round(similarity_ac, 3)
)

print(
    "B vs C:",
    round(similarity_bc, 3)
)


# ============================================================
# 7. INTERPRETATION
# ============================================================

print("\n===== INTERPRETATION =====")

print("\nSentence A:")
print(sentence_a)

print("\nSentence B:")
print(sentence_b)

print("\nSentence C:")
print(sentence_c)


print("\nA and B are about the camera.")

print("A and C are about different features.")

print("B and C are also about different features.")


# ============================================================
# 8. VISUAL CONCEPT
# ============================================================

"""
                    SENTENCE A
                         |
                         |
                    EMBEDDING
                         |
              ----------------------
              |                    |
              ↓                    ↓
        Sentence B           Sentence C
              |                    |
          Camera                  Chip
              |                    |
              ↓                    ↓
        HIGH SIMILARITY       LOWER SIMILARITY


Cosine Similarity measures how similar
two embedding vectors are.

Example:

A ↔ B  = HIGH
A ↔ C  = LOWER
B ↔ C  = LOWER


Pipeline:

Sentence
   ↓
Embedding Model
   ↓
Vector
   ↓
Cosine Similarity
   ↓
Similarity Score
"""