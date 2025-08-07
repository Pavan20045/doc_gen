import os
import pickle
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Load embedding model only once
model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(file_path):
    """Extracts all text from a given PDF file"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_chunks(file_path, chunk_size=500):
    """Splits extracted PDF text into smaller chunks"""
    text = extract_text_from_pdf(file_path)
    print(f" Extracted text length from {file_path}: {len(text)}")
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def embed_text(text):
    """Returns embedding vector of the given text"""
    return model.encode(text)


def build_vector_store():
    """Builds FAISS vector store from uploaded PDFs"""
    pdf_path = "uploads"
    doc_chunks = []
    embedding_dim = 384  # for all-MiniLM-L6-v2

    for filename in os.listdir(pdf_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_path, filename)
            try:
                print(f"📄 Processing: {filename}")
                chunks = extract_text_chunks(filepath)

                for chunk in chunks:
                    if chunk.strip():  # Skip empty chunks
                        embedding = embed_text(chunk)
                        doc_chunks.append((embedding, chunk))
            except Exception as e:
                print(f" Failed to process {filename}: {e}")

    if not doc_chunks:
        raise ValueError("No valid chunks were extracted from PDFs!")

    try:
        embeddings = np.array([item[0] for item in doc_chunks]).astype("float32")

        if len(embeddings.shape) != 2 or embeddings.shape[1] != embedding_dim:
            raise ValueError("Embeddings shape mismatch.")

        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)

        faiss.write_index(index, "projects/faiss_index.index")
        with open("projects/doc_chunks.pkl", "wb") as f:
            pickle.dump(doc_chunks, f)

        print(" FAISS index built and saved.")
    except Exception as e:
        print(f" Error while building FAISS index: {e}")


def retrieve_context(query, top_k=3):
    """Retrieves top-k relevant chunks for a query"""
    try:
        index = faiss.read_index("projects/faiss_index.index")
        with open("projects/doc_chunks.pkl", "rb") as f:
            doc_chunks = pickle.load(f)

        query_embedding = embed_text(query).astype("float32").reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)

        retrieved_chunks = [doc_chunks[i][1] for i in indices[0]]
        return " ".join(retrieved_chunks)
    except Exception as e:
        print(f" Error in retrieve_context: {e}")
        return "Sorry, I couldn't retrieve relevant information."
