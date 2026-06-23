from pydantic import BaseModel
from typing import List

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    status: str

class DocumentItem(BaseModel):
    id: str
    filename: str
    chunks: int
    uploaded_at: str

class DocumentsResponse(BaseModel):
    documents: List[DocumentItem]

class DeleteResponse(BaseModel):
    document_id: str
    status: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ChunkItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[ChunkItem]

class AskRequest(BaseModel):
    query: str
    top_k: int = 5

class SourceItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float

class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]

class HealthResponse(BaseModel):
    status: str
    indexed_documents: int
    memory_usage_mb: float