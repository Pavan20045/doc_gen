import sys
import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load query
input_data = sys.stdin.read()
query_data = json.loads(input_data)
query = query_data["query"]

# Load model + index
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load index and documents
index = faiss.read_index("projects/index.faiss")
with open("projects/docs.pkl", "rb") as f:
    documents = pickle.load(f)

# Embed query
query_embedding = embedder.encode([query])

# Search
D, I = index.search(query_embedding, k=1)

# Get most relevant document
retrieved_context = documents[I[0][0]]

# Send context back to Node.js
print(json.dumps({
    "context": retrieved_context
}))
