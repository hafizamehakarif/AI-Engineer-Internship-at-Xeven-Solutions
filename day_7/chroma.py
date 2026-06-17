# Importing important libraries

import os
import time

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Loading all text documents

all_docs = []

for file in os.listdir("documents"):

    loader = TextLoader(
        f"documents/{file}"
    )

    all_docs.extend(
        loader.load()
    )

print(f"Documents Loaded: {len(all_docs)}")


# Text Splitting

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

print(f"Chunks Created: {len(chunks)}")


# Creating Embeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Creating Chroma Database

start_time = time.time()

vector_store = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory="chroma_db"
)

end_time = time.time()

print("\nChroma DB Created Successfully")

print(
    f"Indexing Time: {end_time - start_time:.2f} seconds"
)


# Similarity Search

query = "What is RAG?"

start_time = time.time()

results = vector_store.similarity_search(
    query,
    k=5
)

end_time = time.time()

print(
    f"Initial Query Time: {end_time - start_time:.4f} seconds"
)

print("\nTop 5 Search Results:\n")

for doc in results:

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    print("-" * 50)


# Reload Chroma Database

print("\nReloading Chroma Database...")

loaded_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

print("Chroma Database Reloaded Successfully")


# Search After Reloading

start_time = time.time()

results = loaded_db.similarity_search(
    "What is artificial intelligence?",
    k=5
)

end_time = time.time()

print(
    f"\nQuery Time After Reloading: {end_time - start_time:.4f} seconds"
)

print("\nSearch Results After Reloading:\n")

for doc in results:

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    print("=" * 50)