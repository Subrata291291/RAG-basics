import nltk
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download("punkt")
nltk.download("punkt_tab")


text = """
The iPhone 15 has a 6.1-inch display.
It uses the A16 Bionic chip.
The phone has a 48MP main camera.
The phone supports 5G connectivity.
The phone comes with Face ID.

The iPhone 15 is available in 128GB, 256GB, and 512GB storage.
The 128GB model costs ₹69,999.
The 256GB model costs ₹79,999.
The 512GB model costs ₹99,999.
The phone is available in five colors.

Customers can return the phone within 7 days.
The product must be unused and in its original packaging.
Refunds are processed within 5-7 business days.
"""


model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = sent_tokenize(text)

print("Total sentences:", len(sentences))

embeddings = model.encode(sentences)


threshold = 0.30

chunks = []
current_chunk = [sentences[0]]


for i in range(1, len(sentences)):

    similarity = cosine_similarity(
        [embeddings[i]],
        [embeddings[i - 1]]
    )[0][0]

    print(
        f"Sentence {i} → Sentence {i + 1} | "
        f"Similarity = {similarity:.3f}"
    )

    if similarity >= threshold:
        current_chunk.append(sentences[i])
    else:
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i]]
if current_chunk:
    chunks.append(" ".join(current_chunk))


print("\n===== SEMANTIC CHUNKS =====")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)



"""
DOCUMENT
   ↓
Split into sentences
   ↓
Sentence 1
Sentence 2
Sentence 3
...
   ↓
Create embedding for every sentence
   ↓
Compare neighboring embeddings
   ↓
Cosine Similarity
   ↓
Compare similarity with threshold
   ↓
HIGH enough → SAME CHUNK
LOW → NEW CHUNK

Result
0.10 ────── ❌ Separate
0.20 ────── ❌ Separate
0.29 ────── ❌ Separate

0.30 ────── ✅ Same
0.40 ────── ✅ Same
0.60 ────── ✅ Same
0.90 ────── ✅ Same

"""