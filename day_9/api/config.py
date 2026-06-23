import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
FAISS_PATH = os.getenv("FAISS_PATH", "./data/faiss_index")
METADATA_PATH = os.getenv("METADATA_PATH", "./data/documents.json")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")