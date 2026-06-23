from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

import os
import json
import shutil


# LOAD ENVIRONMENT VARIABLES

load_dotenv()


# FASTAPI APP


app = FastAPI(title="RAG API")


# CORS

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


# PATHS


DOCUMENTS_FOLDER = "documents"
METADATA_FILE = "documents.json"
FAISS_FOLDER = "faiss_index"

os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)


# EMBEDDINGS


embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# LLM


llm = ChatGroq(
model="llama-3.1-8b-instant",
temperature=0,
api_key=os.getenv("GROQ_API_KEY")
)

# GLOBAL VECTORSTORE


vectorstore = None


# PYDANTIC MODELS

class AskRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str



# METADATA HELPERS


def load_metadata():

    if not os.path.exists(METADATA_FILE):

        with open(METADATA_FILE, "w") as file:
            json.dump([], file)

    with open(METADATA_FILE, "r") as file:
        return json.load(file)


def save_metadata(data):

    with open(METADATA_FILE, "w") as file:
        json.dump(data, file, indent=4)



# HEALTH CHECK


@app.get("/health")
async def health_check():

    documents = load_metadata()
    return {
        "status": "healthy",
        "document_count": len(documents)
    }


# LIST DOCUMENTS


@app.get("/documents")
async def get_documents():
    return load_metadata()


# UPLOAD DOCUMENT

@app.post("/documents/upload")
async def upload_document(
file: UploadFile = File(...)
):

    file_path = os.path.join(
        DOCUMENTS_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

metadata = load_metadata()

document_id = len(metadata) + 1

metadata.append(
    {
        "id": document_id,
        "filename":file.filename
    }
)
save_metadata(metadata) 
return { "message": "Document uploaded successfully", "document_id": document_id }