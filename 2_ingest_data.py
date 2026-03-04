import ray
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import os

# Initialize Ray
ray.init(ignore_reinit_error=True)


# -----------------------------------------------------------
# 1. The MAP Phase (Parallelized CPU Tasks) - Benefit of Ray
# -----------------------------------------------------------
@ray.remote
def process_document(file_path: str):
    print(f"[{file_path}] Loading and chunking...")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    print(f"[{file_path}] Created {len(splits)} semantic chunks.")

    # Filter out "noisy" chunks (e.g., base64 garbage, binary data in text layer)
    clean_splits = []
    for split in splits:
        text = split.page_content
        if len(text) < 100:
            continue

        # Detect garbage: lack of spaces in a long string is a red flag (e.g., base64)
        # Also, very high symbol/digit density can be noisy
        space_count = text.count(" ")
        if space_count / len(text) > 0.1:  # At least 10% spaces
            clean_splits.append(split)
        else:
            print(
                f"[{file_path}] Dropped noisy chunk (Potential base64/garbage). length: {len(text)}"
            )
    return clean_splits


# -----------------------------------------------------------
# 2. Parallel Execution Block
# -----------------------------------------------------------
persist_directory = "./chroma_db"

# -----------------------------------------------------------
# 0. Cleanup Phase (Ensure a fresh database)
# -----------------------------------------------------------
if os.path.exists(persist_directory):
    print(f"Clearing existing database at {persist_directory}...")
    shutil.rmtree(persist_directory)

papers = ["chronos", "mamba"]
print("Dispatching CPU parsing tasks to Ray cluster...")

# Dispatch tasks asynchronously (using Ray for parallel processing)
futures = [process_document.remote(f"./data/{paper}_paper.pdf") for paper in papers]

# Gather all chunks from all workers (Stop parallel processing and wait for all workers to finish chunking)
results = ray.get(futures)

# Flatten the results into a single list of chunks
all_splits = []
for split_list in results:
    all_splits.extend(split_list)

print(f"Successfully aggregated {len(all_splits)} total chunks from all documents.")

# -----------------------------------------------------------
# 3. The REDUCE Phase (Sequential GPU/IO Tasks)
# -----------------------------------------------------------
print("Generating embeddings and safely writing to ChromaDB (Single Thread)...")

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)

# Save all chunks in one single batch here to prevent background workers
# from "clashing" with each other while trying to write to the same file (SQLite locking problem).
Chroma.from_documents(
    documents=all_splits, embedding=embeddings, persist_directory=persist_directory
)

print("All documents embedded and ingested safely!")


# import ray
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma

# # Initialize Ray for local distributed computing
# ray.init()

# @ray.remote
# def process_and_embed_document(file_path: str, persist_dir: str):
#     print("Loading document...")
#     loader = PyPDFLoader(file_path)
#     docs = loader.load()

#     # Deep chunking strategy: preserve paragraphs and overlapping context
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     splits = text_splitter.split_documents(docs)

#     print(f"Created {len(splits)} semantic chunks. Generating embeddings...")

#     # Connect to the local Docker Ollama instance
#     embeddings = OllamaEmbeddings(
#         model="nomic-embed-text",
#         base_url="http://localhost:11434"
#     )

#     # Persist vectors to ChromaDB
#     Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         persist_directory=persist_dir
#     )
#     print("Ingestion complete. Vector database saved.")

# # Execute the Ray task
# persist_directory = "./chroma_db"
# ray.get(process_and_embed_document.remote("./data/chronos_paper.pdf", persist_directory))
