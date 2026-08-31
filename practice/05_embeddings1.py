from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Two sentences
sentence1 = "I love this phone."
sentence2 = "The smartphone is excellent."


# Convert sentences into embeddings
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)


print("Sentence 1:")
print(sentence1)

print("\nSentence 2:")
print(sentence2)


print("\nEmbedding 1:")
print(embedding1)

print("\nEmbedding 2:")
print(embedding2)


# Calculate similarity
similarity = cosine_similarity(
    [embedding1],
    [embedding2]
)[0][0]


print("\nCosine Similarity:")
print(round(similarity, 3))




"""
TEXT
 ↓
EMBEDDING MODEL
 ↓
VECTOR
 ↓
COSINE SIMILARITY
 ↓
HOW SEMANTICALLY SIMILAR?



RAG retrieval:

User Query → embedding

Document chunks → embeddings

       ↓

Compare query embedding
with document embeddings
       ↓
Retrieve most similar chunks
"""