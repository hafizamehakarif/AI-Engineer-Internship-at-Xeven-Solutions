from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Get day_10 directory
BASE_DIR = Path(__file__).resolve().parent
pdf_path = BASE_DIR / "documents" / "fastapi_notes.pdf"


print("PDF Path:", pdf_path)
print("Exists:", pdf_path.exists())

# Load PDF
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Save vector store
vectorstore.save_local(
    str(BASE_DIR / "vectorstore")
)

print("Vector Store Created Successfully!")