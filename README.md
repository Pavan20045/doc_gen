# doc_gen

This project is a web app combining Node.js for the server and Python for document querying using a TinyLLaMA model with FAISS.

## Directory Structure

models/ # Holds your .gguf model file
projects/ # Python scripts for FAISS indexing and querying
public/ # Frontend HTML/CSS assets
uploads/ # Stores user-uploaded files (auto-created)
server.js # Node.js backend
package.json # Node dependencies and scripts
directory_structure.txt # Project skeleton description
web_app_directory.txt # Another descriptor of project layout
.gitignore # Excluded files from version control


---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Pavan20045/doc_gen.git
cd doc_gen
2. Install Node.js Dependencies

npm install
3. Install Python Dependencies
Ensure Python 3.8+ is installed.


pip install faiss-cpu transformers numpy flask huggingface-hub llama-cpp-python
4. Download a GGUF Model from Hugging Face
You can download a GGUF-formatted model (e.g., with Q4_K_M quantization) using the Hugging Face CLI:


pip install huggingface-hub>=0.17.1
huggingface-cli login
huggingface-cli download <username>/<model-repo> <desired-model-filename>.gguf --local-dir models/ --local-dir-use-symlinks False
For example, to download q4 quantization of Microsoft’s Phi‑3 Mini‑4K Instruct model:


huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir models/ --local-dir-use-symlinks False
This step requires authentication with Hugging Face.
Hugging Face

Running the Application
1. Start the Python Backend

python projects/run_llama.py
2. Launch the Node.js Server

node server.js
3. Access in Browser
Open your browser and go to: http://localhost:3000

GGUF Model Usage in Python
Here's how you can load the downloaded .gguf model in your Python scripts using llama-cpp-python:


from llama_cpp import Llama

llm = Llama(
    model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=0  # set >0 if GPU acceleration is available
)

output = llm(
    "<|user|\nYour prompt here<|end|>\n<|assistant|>",
    max_tokens=256,
    stop=["<|end|>"],
    echo=True
)

print(output['choices'][0]['text'])
You can adjust parameters such as n_gpu_layers to enable GPU offloading, or n_ctx for context length.
Hugging Face
+1

Notes
Ensure ports used by your Python and Node.js servers do not conflict.

The uploads/ directory may need write permissions when handling file uploads.

Add large model files to .gitignore to prevent committing them to your Git repo.

Always validate your model prompt formatting to match GGUF expectations.

License
This project is intended for educational and research purposes only.


---

You can copy this entire content into your `README.md`. Let me know if you'd like adjustments—such as adding download examples for different GGUF models or instructions on GPU setup.
::contentReference[oaicite:2]{index=2}
