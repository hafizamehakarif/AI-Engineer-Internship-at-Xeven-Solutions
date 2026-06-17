## Importing important libraries
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#loading the text document 
loader = PyPDFLoader("10 RAG Architectures.pdf")
document = loader.load()
print(document)

#text splitting 
splitter = RecursiveCharacterTextSplitter(

    chunk_size = 250 ,
    chunk_overlap = 50
)
chunk = splitter.split_documents(document)
print(len(chunk))


#create embaddings 
embedding = HuggingFaceEmbeddings(
    model_name ="sentence-transformers/all-MiniLM-L6-v2"
)

#creating vector database
vector_store = FAISS.from_documents(
    chunk, 
    embedding
)

#search 
query = "What is RAGpyth?"

results = vector_store.similarity_search(
    query,
    k=3
)

for doc in results:
    print(doc.page_content)
    print("-" * 50)

vector_store.save_local("faiss_index")

loaded_db = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

results = loaded_db.similarity_search(
    "What is artificial intelligence?",
    k=3
)