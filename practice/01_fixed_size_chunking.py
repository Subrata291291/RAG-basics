text = """
Section 1: Loan Eligibility
The customer must be between 21 and 60 years old.
The minimum salary must be ₹25,000.

Section 2: Disbursal
Loan disbursal occurs within 48 hours.
"""


def fixed_size_chunking(text, chunk_size=10):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


chunks = fixed_size_chunking(text, chunk_size=10)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)



'''
Fixed-size chunking = divide a document into chunks containing a fixed number of words/tokens. 
No embeddings.
No vector database.
No LLM.
No RAG yet.

We're learning one concept at a time.

Chunk 1 → words 1–10
Chunk 2 → words 11–20
'''