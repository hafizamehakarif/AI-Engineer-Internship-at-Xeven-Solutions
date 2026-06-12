# Day 4 LangChain Setup & First Chains

**Intern:** Hafiza Mehak Arif \
**Role:** AI Engineer Intern, Xeven Solutions \
**Date:** June 12, 2026

---

## Overview

Today's focus was on getting started with LangChain and understanding how it can be used to build structured LLM applications. I explored its core concepts and completed the practical tasks using Groq as the LLM provider after resolving the OpenAI API quota issue from the previous day.

---

## Project Structure

```text
day16/
├── README.md
├── LEARNINGS.md
├── langchain.ipynb
├── sample.txt
└── .env (excluded via .gitignore)
```

- **README.md** – Project overview and setup instructions.
- **LEARNINGS.md** – Key concepts, observations, and takeaways.
- **langchain.ipynb** – All implementations for Day 16 tasks.
- **sample.txt** – Sample document used for testing document loaders and Q&A chains.

---

## Setup Instructions

### Install Dependencies

```bash
pip install langchain langchain-community langchain-groq python-dotenv pypdf beautifulsoup4 pandas
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

> `.env` is listed in `.gitignore` and should never be committed.

---

## Tasks Completed

### Task 1: LangChain Setup & First Chain

- Set up LangChain with Groq.
- Tested the `ChatGroq` model wrapper.
- Created reusable prompts using `PromptTemplate`.
- Built the first LCEL chain using:

```python
prompt | model | parser
```

---

### Task 2: Document Loaders

Worked with:

- `TextLoader`
- `PyPDFLoader`
- `WebBaseLoader`
- `CSVLoader`

Explored the `Document` structure containing:

- `page_content`
- `metadata`

Built a generic loader function to handle different file types.

---

### Task 3: Document Q&A Chain

Built a simple document question-answering pipeline that:

- Loaded document content.
- Passed the document and user question to the LLM.
- Returned answers based on the provided document.
- Handled long documents using basic truncation and warning mechanisms.

---

## Concepts Covered

- LangChain and its purpose
- Models, Prompts, Chains, and Memory
- LCEL and the `|` operator
- Prompt templates and output parsers
- Document loaders and the `Document` abstraction
- LangChain vs raw API usage

---

## Status

- ✅ Resolved the OpenAI API quota issue
- ✅ Set up LangChain with Groq
- ✅ Built the first LCEL chain
- ✅ Practiced document loaders
- ✅ Implemented a document Q&A chain
- ✅ Documented learnings and implementations
