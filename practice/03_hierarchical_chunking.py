import re


text = """
Section 1: Loan Eligibility
The customer must be between 21 and 60 years old.
The minimum salary must be ₹25,000.
The credit score must be above 700.

Section 2: Disbursal
Loan disbursal occurs within 48 hours.
Delays can occur due to incomplete documentation.

Section 3: EMI Defaults
Missing two or more EMIs may result in penalties.
Further defaults may lead to legal action.

Section 4: Foreclosure
Customers can foreclose their loan after 6 EMIs.
A foreclosure fee of 2% is applicable.
"""


def hierarchical_chunking(text):
    sections = re.split(r'\n(?=Section \d+:)', text.strip())

    chunks = []

    for section in sections:
        section = section.strip()

        if section:
            chunks.append(section)

    return chunks


chunks = hierarchical_chunking(text)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)



"""
Fixed-size: "Every 10 words → new chunk."

Hierarchical: "Every new Section → new chunk."

"""