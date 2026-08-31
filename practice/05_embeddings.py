from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Text
text = "The iPhone 15 has a 48MP camera."


# Convert text into embedding
embedding = model.encode(text)


print("Text:")
print(text)

print("\nEmbedding:")
print(embedding)

print("\nEmbedding dimensions:")
print(len(embedding))