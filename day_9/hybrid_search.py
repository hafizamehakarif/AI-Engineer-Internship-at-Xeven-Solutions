from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  


#load documents 
# Load documents
loader = PyPDFDirectoryLoader("documents/")
documents = loader.load()

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Documents split into {len(chunks)} chunks.")
#create embeddings 

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
print("Embeddings created successfully!")

#creating Faiss vector store 
vectorstore = FAISS.from_documents(chunks,embeddings)

print("FAISS vector store created successfully!")


faiss_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 20}
)

bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = 20

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        faiss_retriever,
        bm25_retriever
    ],
    weights=[
        0.7,
        0.3
    ]
)
print("Welcome to the Hybrid Search Engine!")

while True:

    query = input(
        "Please enter your query (or type 'exit' to quit): "
    )


    if query.lower() == "exit":
        break

    results = hybrid_retriever.invoke(query)

    print("Top 3 Relevant Chunks:")

    for i, result in enumerate(results, 1):

        print(f"{i}. {result.page_content}")
        print("-" * 50)