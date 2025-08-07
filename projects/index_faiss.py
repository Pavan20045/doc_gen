import os
import pickle
import faiss
from vector_store import embed_text, extract_text_chunks

pdf_path = "uploads"
doc_chunks = []
embedding_dim = 384  # make sure this matches your embedding size

for filename in os.listdir(pdf_path):
    if filename.endswith(".pdf"):
        filepath = os.path.join(pdf_path, filename)
        try:
            print(f"Parsing: {filename}")
            chunks = extract_text_chunks(filepath)

            for chunk in chunks:
                if chunk.strip():  # skip empty strings
                    embedding = embed_text(chunk)
                    doc_chunks.append((embedding, chunk))
        except Exception as e:
            print(f" Failed to process {filename}: {e}")

# Check before building FAISS index
if not doc_chunks:
    raise ValueError("No valid chunks were extracted from PDFs!")

# Build FAISS index
try:
    dimension = embedding_dim
    index = faiss.IndexFlatL2(dimension)
    embeddings = [item[0] for item in doc_chunks]

    # Make sure all embeddings are correct shape
    import numpy as np
    embeddings = np.array(embeddings).astype("float32")
    if len(embeddings.shape) != 2 or embeddings.shape[1] != dimension:
        raise ValueError("Embeddings shape mismatch.")

    index.add(embeddings)

    faiss.write_index(index, "projects/faiss_index.index")
    with open("projects/doc_chunks.pkl", "wb") as f:
        pickle.dump(doc_chunks, f)

    print("FAISS index built and saved.")
except Exception as e:
    print(f" Error while building FAISS index: {e}")
