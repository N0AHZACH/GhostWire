from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read knowledge file
with open("rag_system/knowledge/ai.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into sentences
sentences = text.split("\n")

# Create embeddings
sentence_embeddings = model.encode(sentences)

# Ask user query
query = input("Enter claim or question: ")

# Convert query to embedding
query_embedding = model.encode([query])

# Compute similarity
scores = cosine_similarity(query_embedding, sentence_embeddings)

# Find best matching sentence
best_index = scores.argmax()

print("\nEvidence Found:\n")
print(sentences[best_index])