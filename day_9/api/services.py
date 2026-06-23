import os
import json
import uuid
from datetime import datetime
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI
from fastapi import UploadFile, HTTPException

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    FAISS_PATH,
    METADATA_PATH,
)

llm_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

emb_client = OpenAI(api_key=OPENAI_API_KEY)

os.makedirs(FAISS_PATH, exist_ok=True)
os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

INDEX_FILE = os.path.join(FAISS_PATH, "index.bin")

def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    resp = emb_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype="float32")

def load_metadata() -> list:
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_metadata(data: list):
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_index(dim: int):
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    return faiss.IndexFlatL2(dim)

def save_index(index):
    faiss.write_index(index, INDEX_FILE)

async def read_text_file(file: UploadFile) -> str:
    raw = await file.read()
    try:
        return raw.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Only UTF-8 text files are supported.")

async def add_document(file: UploadFile) -> Tuple[str, str, int]:
    text = await read_text_file(file)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty file.")

    doc_id = str(uuid.uuid4())
    vectors = embed_texts(chunks)

    index = load_index(vectors.shape[1])
    start_pos = index.ntotal
    index.add(vectors)

    metadata = load_metadata()
    now = datetime.utcnow().isoformat()

    for i, chunk in enumerate(chunks):
        metadata.append({
            "chunk_id": f"{doc_id}_{i}",
            "document_id": doc_id,
            "filename": file.filename,
            "chunk_index": i,
            "vector_id": start_pos + i,
            "text": chunk,
            "uploaded_at": now,
        })

    save_index(index)
    save_metadata(metadata)
    return doc_id, file.filename, len(chunks)

def list_documents():
    metadata = load_metadata()
    docs = {}
    for item in metadata:
        d_id = item["document_id"]
        if d_id not in docs:
            docs[d_id] = {
                "id": d_id,
                "filename": item["filename"],
                "chunks": 0,
                "uploaded_at": item["uploaded_at"],
            }
        docs[d_id]["chunks"] += 1
    return list(docs.values())

def delete_document(document_id: str):
    metadata = load_metadata()
    remaining = [m for m in metadata if m["document_id"] != document_id]
    if len(remaining) == len(metadata):
        raise HTTPException(status_code=404, detail="Document not found")
    save_metadata(remaining)
    return True

def search_chunks(query: str, top_k: int = 5):
    metadata = load_metadata()
    if not metadata or not os.path.exists(INDEX_FILE):
        return []

    q_vec = embed_texts([query])
    index = faiss.read_index(INDEX_FILE)
    distances, indices = index.search(q_vec, top_k)

    by_vector_id = {m["vector_id"]: m for m in metadata}
    results = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1 or idx not in by_vector_id:
            continue
        item = by_vector_id[idx]
        results.append({
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "text": item["text"],
            "score": float(dist),
        })

    return results

def ask_question(query: str, top_k: int = 5):
    chunks = search_chunks(query, top_k)
    context = "\n\n".join([c["text"] for c in chunks])

    prompt = f"""Use only the context below to answer.

Context:
{context}

Question:
{query}
"""

    resp = llm_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Answer only from the provided context."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = resp.choices[0].message.content
    return answer, chunks

def health_status():
    metadata = load_metadata()
    return len(set(m["document_id"] for m in metadata))