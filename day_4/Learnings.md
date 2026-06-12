# Day 4 LangChain Setup & First Chains

**Intern:** Hafiza Mehak Arif \
**Date:** June 11, 2026 \
**Topic:** Introduction to LangChain, LCEL, Document Loaders

---

## What I Learned

### 1. What is LangChain?

LangChain is a framework that abstracts away the repetitive boilerplate
of working with LLM APIs directly. Instead of manually handling prompts,
API calls, and response parsing every time, LangChain provides reusable,
composable building blocks.

### 2. Core Components

- **Models** — standardized wrappers around LLM providers (OpenAI,
  Anthropic, Google), making it easy to swap models.
- **Prompts** — reusable templates with variables (`PromptTemplate`).
- **Chains** — sequences of steps where output of one feeds into the next.
- **Memory** — allows chains to retain context across interactions
  (covered in more depth later in the roadmap).

### 3. LCEL (LangChain Expression Language)

- Uses the pipe operator `|` to connect components:
  `chain = prompt | model | output_parser`
- Every component implements the **Runnable** interface, supporting
  `.invoke()`, `.stream()`, and `.batch()`.
- This shared interface is _why_ piping works — all pieces "speak the
  same language."

### 4. Document Loaders

- Convert different file types (text, PDF, CSV, web pages) into a
  standardized `Document` object:
  ```
  Document(page_content="...", metadata={"source": "..."})
  ```
- Because every loader returns this same structure, downstream steps
  (chunking, embedding, RAG) work identically regardless of source.

### 5. LangChain vs Raw APIs

- Raw APIs: best for single, simple calls.
- LangChain: best for multi-step workflows, document processing,
  and agents — which is the direction of this internship (RAG, Week 4).

---

## What I Built

- **Task 1:** Set up `ChatOpenAI` model wrapper, created a
  `PromptTemplate` with variables, and built my first LCEL chain
  (`prompt | model | output_parser`).
- **Task 2:** Practiced loading documents using `TextLoader`,
  `PyPDFLoader`, `WebBaseLoader`, and `CSVLoader`. Built a generic
  loader function that detects file type and selects the right loader.
- **Task 3:** Built a Document Q&A chain — loads a document, formats
  it into a prompt with the user's question, and returns an answer
  from the model.

---

## Key Insight

A chain like `prompt | model | output_parser` replaces three manual
steps (format prompt → call API → extract text) with one composable
pipeline. This pattern is the foundation for everything coming next —
RAG pipelines, agents, and production APIs all build on this same
chaining concept.

---

## References

1. **Article:** "LangChain Expression Language Explained" — Pinecone
   https://www.pinecone.io/learn/series/langchain/langchain-expression-language/
2. **Article:** "LangChain Document Loaders: A Guide to PDF, CSV, &
   Web Ingestion" — Medium
   https://medium.com/@ashutoshsharmaengg/feeding-your-llm-right-mastering-langchains-document-loaders-64ff06675c7b

---
