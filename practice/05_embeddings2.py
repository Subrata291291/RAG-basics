from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Three sentences
sentence_a = "I love this phone."
sentence_b = "The smartphone is excellent."
sentence_c = "The weather is very cold today."


# Create embeddings
embedding_a = model.encode(sentence_a)
embedding_b = model.encode(sentence_b)
embedding_c = model.encode(sentence_c)


# Compare A with B
similarity_ab = cosine_similarity(
    [embedding_a],
    [embedding_b]
)[0][0]


# Compare A with C
similarity_ac = cosine_similarity(
    [embedding_a],
    [embedding_c]
)[0][0]


# Compare B with C
similarity_bc = cosine_similarity(
    [embedding_b],
    [embedding_c]
)[0][0]


print("A:", sentence_a)
print("B:", sentence_b)
print("C:", sentence_c)

print("\n===== SIMILARITY =====")

print("A vs B:", round(similarity_ab, 3))
print("A vs C:", round(similarity_ac, 3))
print("B vs C:", round(similarity_bc, 3))



"""
                 PHONE
                  ● A
                ↗
              ↗
            ● B
       smartphone


                         WEATHER
                            ● C

A and B are closer in semantic space.
C is far away.

The embedding model converts language into a numerical representation where related meanings tend to be closer together.



USER QUERY
    ↓
"What camera does the iPhone 15 have?"
    ↓
Create embedding
    ↓
Compare against every document embedding
    ↓
Calculate cosine similarity
    ↓
        ┌───────────────┐
        ↓               ↓
   Document 1       Document 2
   similarity       similarity
     HIGH             LOW
        ↓
    Document 1
        ↓
   RETRIEVE IT
"""