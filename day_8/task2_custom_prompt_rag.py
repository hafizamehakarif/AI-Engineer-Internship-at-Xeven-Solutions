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

load_dotenv()

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
# CHUNK DOCUMENTS
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
# RETRIEVER
# Uses a similarity score threshold so a query with no genuinely
# relevant chunks returns an EMPTY list instead of forcing back
# the top-k unrelated ones anyway.
# ==========================
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.4},
)

# ==========================
# GROQ LLM
# ==========================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
)

# ==========================
# STEP 1: PROMPT TEMPLATE
# ==========================
prompt = PromptTemplate.from_template(
    """Answer based ONLY on context.
If answer not in context, say "I don't have that information".

Context: {context}
Question: {question}

Answer:"""
)

# ==========================
# STEP 2: FORMAT CONTEXT
# ==========================
def format_context(docs):
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================
# STEP 3 & 4: CHAIN
# retriever -> format_context -> prompt -> llm -> parse
# ==========================
rag_chain = (
    {"context": retriever | format_context, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================
# LIGHTWEIGHT FOLLOW-UP HANDLING
# Plain RAG has no memory, so "what about X" on its own retrieves
# nothing useful. This folds the last Q&A into the search query
# before retrieval so follow-ups have enough context to match.
# ==========================
chat_history = []  # list of (question, answer)

def rewrite_with_history(question):
    if not chat_history:
        return question
    last_q, last_a = chat_history[-1]
    return f"Previous question: '{last_q}'. Previous answer: '{last_a}'. Follow-up: {question}"

def ask(question, use_history=True):
    search_question = rewrite_with_history(question) if use_history else question
    retrieved_docs = retriever.invoke(search_question)
    answer = rag_chain.invoke(search_question)
    chat_history.append((question, answer))
    return answer, retrieved_docs

# ==========================
# EDGE CASE TESTS
# ==========================
def run_edge_case_tests():
    print("\n" + "=" * 60)
    print("RUNNING EDGE CASE TESTS")
    print("=" * 60)

    print("\n[No relevant docs] Question: What is the capital of Mars?")
    answer, docs = ask("What is the capital of Mars?", use_history=False)
    print(f"Answer: {answer}")
    print(f"Docs Retrieved: {len(docs)}")

    print("\n[Ambiguous question] Question: What about the policy?")
    answer, docs = ask("What about the policy?", use_history=False)
    print(f"Answer: {answer}")
    print(f"Docs Retrieved: {len(docs)}")

    print("\n[Follow-up] Question: Can you explain that in more detail?")
    answer, docs = ask("Can you explain that in more detail?")
    print(f"Answer: {answer}")
    print(f"Docs Retrieved: {len(docs)}")

# ==========================
# QUESTION LOOP
# ==========================
if __name__ == "__main__":
    mode = input("Type 'test' to run edge case tests, or press Enter to chat: ").strip().lower()

    if mode == "test":
        run_edge_case_tests()
    else:
        while True:
            question = input("\nAsk a Question (type exit to quit): ")
            if question.lower() == "exit":
                break

            answer, retrieved_docs = ask(question)

            print("\nAnswer:")
            print(answer)

            print("\nSources Used:")
            if not retrieved_docs:
                print("None (no documents matched this query above the similarity threshold)")
            else:
                for i, doc in enumerate(retrieved_docs, start=1):
                    print(f"{i}. {doc.metadata.get('source', 'Unknown Source')}")

            print("\n" + "=" * 60)