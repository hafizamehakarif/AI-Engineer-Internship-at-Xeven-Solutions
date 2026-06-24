import os
import shutil

from fastapi import FastAPI, UploadFile, File
from rag import process_pdf, ask_question

app = FastAPI(
    title="Document RAG API",
    version="1.0"
)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#home route
@app.get("/")
def home():
    return {
        "message": "Document RAG API is Running"
    }

#upload route
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_pdf(file_path)

    return {
        "message": f"{file.filename} uploaded successfully"
    }

#query route
@app.post("/query")
def query_document(question: str):

    answer = ask_question(question)

    return {
        "question": question,
        "answer": answer
    }
# health check route
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
# get documents route
@app.get("/documents")
def get_documents():

    files = os.listdir(UPLOAD_FOLDER)

    return {
        "documents": files
    }
    # delete document route
@app.delete("/documents/{filename}")
def delete_document(filename: str):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    if not os.path.exists(file_path):
        return {
            "message": "Document not found"
        }

    os.remove(file_path)

    return {
        "message": f"{filename} deleted successfully"
    }