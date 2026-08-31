text = """
Section 1: Loan Eligibility
The customer must be between 21 and 60 years old.
The minimum salary must be ₹25,000.

Section 2: Disbursal
Loan disbursal occurs within 48 hours.
"""


def overlap_chunking(text, chunk_size=10, overlap=3):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start = start + chunk_size - overlap

    return chunks


chunks = overlap_chunking(text, chunk_size=10, overlap=3)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)


"""
Some words will be repeated.

That's the whole idea of overlap.

Chunk 1 → words 1–10
Chunk 2 → words 8–17
Chunk 3 → words 15–24

"""