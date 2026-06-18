import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

# ==========================
# WEBSITE SOURCES
# ==========================
WEB_URLS = [
    ("https://docs.langchain.com/oss/python/langchain/knowledge-base", "langchain"),
    ("https://docs.python.org/3/", "Python")
]

# ==========================
# LOAD LOCAL DOCUMENTS
# ==========================
def load_local_documents(root="documents"):
    docs = []
    if not os.path.exists(root):
        print(f"⚠️  Warning: '{root}' folder not found. Creating empty directory...")
        os.makedirs(root)
        return docs
    
    for domain in os.listdir(root):
        domain_path = os.path.join(root, domain)
        if not os.path.isdir(domain_path):
            continue

        for file in os.listdir(domain_path):
            file_path = os.path.join(domain_path, file)

            if file.endswith(".txt"):
                loader = TextLoader(file_path)
                source_type = "txt"
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                source_type = "pdf"
            else:
                continue

            file_docs = loader.load()
            mod_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d")

            for doc in file_docs:
                doc.metadata.update({
                    "source_type": source_type,
                    "domain": domain,
                    "date": mod_date,
                    "source": file_path,
                })
            docs.extend(file_docs)
    return docs

# ==========================
# LOAD WEBSITES
# ==========================
def load_web_documents(urls_with_domains):
    docs = []
    for url, domain in urls_with_domains:
        loader = WebBaseLoader(url)
        web_docs = loader.load()
        scrape_date = datetime.now().strftime("%Y-%m-%d")

        for doc in web_docs:
            doc.metadata.update({
                "source_type": "web",
                "domain": domain,
                "date": scrape_date,
                "source": url,
            })
        docs.extend(web_docs)
    return docs

# ==========================
# LOAD ALL DOCUMENTS
# ==========================
all_docs = load_local_documents("documents") + load_web_documents(WEB_URLS)
print(f"\n✅ Documents Loaded: {len(all_docs)}")

if len(all_docs) == 0:
    print("⚠️  No documents found! Add .txt or .pdf files to 'documents/domain/' folder")

# ==========================
# CHUNK DOCUMENTS
# ==========================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_docs)
print(f"📑 Chunks Created: {len(chunks)}")

# ==========================
# EMBEDDINGS + FAISS
# ==========================
print("\n🔄 Loading embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    print("📂 Loading existing FAISS index...")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("💾 Creating new FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("✅ FAISS Index Saved Successfully")

# ==========================
# FILTERED RETRIEVAL
# ==========================
def retrieve_filtered(question, k=4, source_type=None, domain=None):
    filter_dict = {}
    if source_type:
        filter_dict["source_type"] = source_type
    if domain:
        filter_dict["domain"] = domain

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
        filter=filter_dict if filter_dict else None,
    )
    
    return [
        {"doc": doc, "relevance": round(1 / (1 + score), 4)}
        for doc, score in results
    ]

# ==========================
# CONTEXT FORMATTING
# ==========================
def format_context(scored_chunks):
    if not scored_chunks:
        return ""
    parts = []
    for item in scored_chunks:
        doc = item["doc"]
        tag = f"[{doc.metadata.get('source_type', 'unknown')} | {doc.metadata.get('domain', 'unknown')}]"
        parts.append(f"{tag}\n{doc.page_content}")
    return "\n\n".join(parts)

# ==========================
# PROMPTS + LLM
# ==========================
prompt = PromptTemplate.from_template(
    """Answer based ONLY on context.
If answer not in context, say "I don't have that information".
If the context contains information from multiple sources, synthesize
them into one coherent answer rather than listing them separately.

Context: {context}
Question: {question}

Answer:"""
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY", ""),
)
output_parser = StrOutputParser()

# ==========================
# SESSION HISTORY
# ==========================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ==========================
# FOLLOW-UP QUESTION HANDLING
# ==========================
contextualize_prompt = PromptTemplate.from_template(
    """
Given the chat history and a follow-up question,
rewrite the follow-up into a standalone question.

Chat History:
{chat_history}

Question:
{question}

Standalone Question:
"""
)

contextualize_chain = (
    contextualize_prompt
    | llm
    | StrOutputParser()
)

def format_chat_history(messages):
    return "\n".join(
        f"{msg.type}: {msg.content}"
        for msg in messages
    )

# ==========================
# FULL QUERY FUNCTION (FIXED!)
# ==========================
def query(
    question,
    session_id="default",
    source_type=None,
    domain=None,
    k=4
):
    history = get_session_history(session_id)
    
    # Handle follow-up questions
    if history.messages:
        standalone_question = contextualize_chain.invoke(
            {
                "chat_history": format_chat_history(history.messages),
                "question": question,
            }
        )
    else:
        standalone_question = question
    
    # RETRIEVE (always do this)
    scored_chunks = retrieve_filtered(standalone_question, k=k, source_type=source_type, domain=domain)
    context = format_context(scored_chunks)

    # GENERATE answer
    chain_input = prompt.format(context=context, question=standalone_question)
    raw_answer = llm.invoke(chain_input)
    answer = output_parser.invoke(raw_answer)
    
    # Add to history
    history.add_user_message(question)
    history.add_ai_message(answer)

    # Get sources
    sources = [
        {
            "source": item["doc"].metadata.get("source", "Unknown"),
            "source_type": item["doc"].metadata.get("source_type", "Unknown"),
            "domain": item["doc"].metadata.get("domain", "Unknown"),
            "date": item["doc"].metadata.get("date", "Unknown"),
            "relevance_score": item["relevance"],
        }
        for item in scored_chunks
    ]
    
    return {"answer": answer, "sources": sources}

def print_result(result):
    print("\n" + "=" * 60)
    print("🤖 Answer:")
    print(result["answer"])

    print("\n📚 Sources Used:")
    if not result["sources"]:
        print("   None")
    else:
        for i, s in enumerate(result["sources"], start=1):
            print(
                f"   {i}. {s['source']} "
                f"(type={s['source_type']}, domain={s['domain']}, "
                f"date={s['date']}, relevance={s['relevance_score']})"
            )
    print("=" * 60)

# ==========================
# QUERY LOOP
# ==========================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎯 RAG Question Answering System")
    print("=" * 60)
    print("Leave filters blank to search across all sources.")
    print("Type 'exit' to quit")
    print("=" * 60)
    
    while True:
        question = input("\nAsk a Question: ").strip()
        if question.lower() == "exit":
            break

        source_type = input("Filter by source_type (pdf/txt/web or blank): ").strip() or None
        domain = input("Filter by domain (or blank): ").strip() or None

        result = query(question, source_type=source_type, domain=domain)
        print_result(result)