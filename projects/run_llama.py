import sys
import json
from llama_cpp import Llama
from vector_store import retrieve_context

# Get the question from command-line args
query = sys.argv[1]

# Load context from FAISS
document_text = retrieve_context(query)

# === FIX: Limit document_text to avoid context overflow ===
MAX_DOC_WORDS = 1500  # adjust if needed

def truncate_text(text, max_words=MAX_DOC_WORDS):
    return ' '.join(text.split()[:max_words])

document_text = truncate_text(document_text)

# === Load LLaMA model ===
llm = Llama(
    model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=0,
    chat_format="chatml"
)

# === Create chat completion ===
response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the indexed document."
        },
        {
            "role": "user",
            "content": f"{document_text}\n\nQuestion: {query}"
        }
    ],
    max_tokens=512  # Ensure room for output
)

# === Output answer as JSON ===
print(json.dumps({
    "question": query,
    "answer": response["choices"][0]["message"]["content"]
}))
