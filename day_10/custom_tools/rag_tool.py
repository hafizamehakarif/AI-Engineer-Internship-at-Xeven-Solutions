from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

vectorstore = FAISS.load_local(
    str(BASE_DIR / "vectorstore"),
    embeddings,
    allow_dangerous_deserialization=True
)
rag_search_calls = 0
@tool
def rag_search(query: str) -> str:
    """
    Search the document index and return relevant information.
    """
    global rag_search_calls
    rag_search_calls += 1
    try:
        docs = vectorstore.similarity_search(
            query=query,
            k=3
        )

        if not docs:
            return "No relevant information found."

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    except Exception as e:
        return f"RAG Error: {e}"

