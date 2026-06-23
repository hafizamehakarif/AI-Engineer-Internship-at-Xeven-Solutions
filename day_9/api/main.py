import time
import psutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS
from schemas import (
    UploadResponse, DocumentsResponse, DeleteResponse,
    SearchRequest, SearchResponse, AskRequest, AskResponse,
    HealthResponse, DocumentItem, ChunkItem, SourceItem
)
from services import add_document, list_documents, delete_document, search_chunks, ask_question, health_status

app = FastAPI(title="Simple RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start
        print(f"{request.method} {request.url.path} -> {response.status_code} in {duration:.3f}s")
        return response
    except Exception as e:
        duration = time.time() - start
        print(f"ERROR {request.method} {request.url.path} after {duration:.3f}s: {e}")
        raise

@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        doc_id, filename, chunks = await add_document(file)
        return UploadResponse(document_id=doc_id, filename=filename, chunks=chunks, status="indexed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=DocumentsResponse)
async def get_documents():
    try:
        docs = list_documents()
        return DocumentsResponse(documents=[DocumentItem(**d) for d in docs])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{id}", response_model=DeleteResponse)
async def remove_document(id: str):
    try:
        delete_document(id)
        return DeleteResponse(document_id=id, status="deleted")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    try:
        results = search_chunks(req.query, req.top_k)
        return SearchResponse(query=req.query, results=[ChunkItem(**r) for r in results])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        answer, chunks = ask_question(req.query, req.top_k)
        return AskResponse(
            query=req.query,
            answer=answer,
            sources=[SourceItem(**c) for c in chunks]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health():
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    return HealthResponse(
        status="ok",
        indexed_documents=health_status(),
        memory_usage_mb=round(mem, 2),
    )