# RAG LLM: Research Paper Explorer

A modular Retrieval Augmented Generation (RAG) pipeline built with **Ray**, **LangChain**, and **Ollama**. This project allows you to download, ingest, and chat with AI research papers (like Mamba and Chronos) using a local Mistral model.

## Key Features

- **Distributed Data Ingestion**: Uses Ray to parallelize PDF parsing and text chunking.
- **Local LLM & Embeddings**: Powered by Ollama (Mistral & Nomic-Embed-Text) for 100% local privacy.
- **Production-Ready Serving**: Uses Ray Serve and FastAPI for the inference backend.
- **Interactive UI**: A Streamlit chat interface with A/B testing (RAG vs. Direct LLM).

---

## File Breakdown

This project follows a 4-step sequential pipeline:

### 1. [1_download_data.py]
**Purpose**: Parallel high-speed data acquisition.
- Uses **Ray** to dispatch multiple download tasks simultaneously.
- Downloads target research papers (e.g., Mamba, Chronos) from ArXiv to the local `./data` directory.

### 2. [2_ingest_data.py]
**Purpose**: Building the "Search Intelligence" (Vector Database).
- **Parsing**: Uses `PyMuPDFLoader` to extract text from PDFs.
- **Parallel Processing**: Ray handles the CPU-intensive task of splitting and filtering the text into semantic chunks.
- **Cleaning**: Automatically filters out "garbage" or binary data often found in PDF layers.
- **Embedding & Storage**: Generates vector embeddings via Ollama and saves them into a local **ChromaDB** instance (`./chroma_db`).

### 3. [3_serve_rag.py]
**Purpose**: The Backend Brain (Inference Service).
- Deploys a **Ray Serve** actor that integrates **FastAPI**.
- **Service Logic**: Combines the ChromaDB retriever with the Mistral LLM.
- **API Endpoint**: Exposes a `/rag` endpoint that receives queries and returns contextualized answers with source citations.
- Supports a `use_rag` toggle to compare AI performance with and without retrieved context.

### 4. [4_streamlit_app.py]
**Purpose**: The User Interface.
- Provides a polished, easy-to-use chat box.
- Features a **Pipeline Control** sidebar to toggle RAG on/off in real-time.
- Displays the LLM's answers alongside verified source page numbers from the ingested documents.

---

## Setup & Installation

### 1. Prerequisites
- **Ollama**: [Download Ollama](https://ollama.ai/) and pull the required models:
  ```bash
  ollama pull mistral
  ollama pull nomic-embed-text
  ```
- **Python 3.10+**

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv rag_env
source rag_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run the Pipeline

Follow these steps in order:

### Step 1: Download Data
```bash
python 1_download_data.py
```

### Step 2: Ingest into Vector DB
Ensure Ollama is running, then:
```bash
python 2_ingest_data.py
```

### Step 3: Start the Backend Service
```bash
serve run 3_serve_rag:rag_app
```

### Step 4: Launch the UI
In a new terminal (with the virtual env activated):
```bash
streamlit run 4_streamlit_app.py
```

---

## ⚙️ Configuration
- **Models**: You can change the models in `3_serve_rag.py` (default: `mistral`) and `2_ingest_data.py` (default: `nomic-embed-text`).
- **Chunking**: Adjust `chunk_size` and `chunk_overlap` in `2_ingest_data.py` to tune retrieval accuracy.
- **Ray Cluster**: By default, Ray runs locally on all available cores.


