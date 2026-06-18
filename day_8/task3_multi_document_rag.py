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

load_dotenv()

# ==========================
# WEBSITE SOURCES (url, domain_label)
# ==========================
WEB_URLS = [
    ("https://docs.langchain.com/oss/python/langchain/knowledge-base", "langchain"),
    ("https://docs.python.org/3/","Python")
]

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
    all_docs.extend(loader.load())

print(f"\nDocuments Loaded: {len(all_docs)}")
# ==========================
def load_local_documents(root="documents"):
    docs = []
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
# LOAD + TAG: WEBSITES
# Requires: pip install beautifulsoup4
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

all_docs = load_local_documents("documents") + load_web_documents(WEB_URLS)
print(f"\nDocuments Loaded: {len(all_docs)}")

# ==========================
# CHUNK DOCUMENTS
# Metadata carries over to every chunk automatically.
# ==========================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_docs)
print(f"Chunks Created: {len(chunks)}")

# ==========================
# EMBEDDINGS + FAISS
# ==========================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
print("FAISS Index Saved Successfully")

# ==========================
# FILTERED RETRIEVAL
# FAISS's `filter` arg does exact-match on metadata fields.
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
    # FAISS returns L2 distance (lower = more similar) — convert to a
    # 0-1 relevance proxy for display. This is approximate, not a calibrated probability.
    return [
        {"doc": doc, "relevance": round(1 / (1 + score), 4)}
        for doc, score in results
    ]

# ==========================
# CONTEXT FORMATTING (tags each chunk by source for the LLM)
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
# SYNTHESIS PROMPT
# Explicitly asks the LLM to merge info from multiple chunks/sources.
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
    api_key=os.environ["GROQ_API_KEY"],
)
output_parser = StrOutputParser()

# ==========================
# FULL QUERY FUNCTION
# Returns: {"answer": ..., "sources": [{source, source_type, domain, date, relevance_score}]}
# ==========================
def query(question, source_type=None, domain=None, k=4):
    scored_chunks = retrieve_filtered(question, k=k, source_type=source_type, domain=domain)
    context = format_context(scored_chunks)

    chain_input = prompt.format(context=context, question=question)
    raw_answer = llm.invoke(chain_input)
    answer = output_parser.invoke(raw_answer)

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
    print("\nAnswer:")
    print(result["answer"])

    print("\nSources Used:")
    if not result["sources"]:
        print("None")
    else:
        for i, s in enumerate(result["sources"], start=1):
            print(
                f"{i}. {s['source']} "
                f"(type={s['source_type']}, domain={s['domain']}, "
                f"date={s['date']}, relevance={s['relevance_score']})"
            )

# ==========================
# QUERY LOOP
# ==========================
if __name__ == "__main__":
    print("\nLeave filters blank to search across all sources.")
    while True:
        question = input("\nAsk a Question (type exit to quit): ")
        if question.lower() == "exit":
            break

        source_type = input("Filter by source_type (pdf/txt/web or blank): ").strip() or None
        domain = input("Filter by domain (or blank): ").strip() or None

        result = query(question, source_type=source_type, domain=domain)
        print_result(result)
        print("\n" + "=" * 60)