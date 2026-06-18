import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================
# LOAD DOCUMENTS
# ==========================
all_docs = []
for file in os.listdir("documents"):
    file_path = os.path.join("documents", file)

    if file.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        continue

    docs = loader.load()
    all_docs.extend(docs)

print(f"\nDocuments Loaded: {len(all_docs)}")

# ==========================
# CHUNK DOCUMENTS
# ==========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(all_docs)
print(f"Chunks Created: {len(chunks)}")

# ==========================
# EMBEDDINGS
# ==========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# FAISS VECTOR STORE
# ==========================
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
print("FAISS Index Saved Successfully")

# ==========================
# RETRIEVER
# ==========================
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ==========================
# GROQ LLM
load_dotenv()  # reads .env file in the current directory and loads variables into os.environ

# ... other imports and code ...

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"]
)
# ==========================
# PROMPT TEMPLATE
# ==========================
prompt = PromptTemplate.from_template(
    """You are a helpful AI assistant.
Answer ONLY from the provided context.
If the answer is not available in the context, say:
"I don't have that information in the provided documents."

Context: {context}

Question: {question}

Answer:"""
)

# ==========================
# FORMAT DOCUMENTS
# ==========================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================
# RAG CHAIN
# ==========================
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================
# QUESTION LOOP
# ==========================
while True:
    question = input("\nAsk a Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    retrieved_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    print("\nAnswer:")
    print(answer)

    print("\nSources Used:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"{i}. {doc.metadata.get('source', 'Unknown Source')}")

    print("\n" + "=" * 60)