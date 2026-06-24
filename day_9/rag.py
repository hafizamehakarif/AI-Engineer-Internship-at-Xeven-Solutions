import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

VECTOR_DB_PATH = "vector_store"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)
def process_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    faiss_file = os.path.join(
        VECTOR_DB_PATH,
        "index.faiss"
    )

    if os.path.exists(faiss_file):

        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        vectorstore.add_documents(chunks)

    else:

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    vectorstore.save_local(VECTOR_DB_PATH)

    return "Document Stored Successfully"

def ask_question(question):

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content