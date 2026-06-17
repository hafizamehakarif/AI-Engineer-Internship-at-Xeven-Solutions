# Importing important libraries

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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


# Creating Vector Database

vector_store = FAISS.from_documents(
    chunks,
    embedding
)

print("FAISS Vector Store Created Successfully")


# Similarity Search

query = "What is RAG?"

results = vector_store.similarity_search(
    query,
    k=5
)

print("\nTop 5 Search Results:\n")

for doc in results:

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    print("-" * 50)


# Save FAISS Index

vector_store.save_local("faiss_index")

print("\nFAISS Index Saved Successfully")


# Load FAISS Index

loaded_db = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

print("FAISS Index Loaded Successfully")


# Search After Reloading

results = loaded_db.similarity_search(
    "What is artificial intelligence?",
    k=5
)

print("\nSearch Results After Reloading Index:\n")

for doc in results:

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    print("=" * 50)