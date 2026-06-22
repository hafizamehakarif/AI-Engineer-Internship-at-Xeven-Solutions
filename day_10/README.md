# LangChain Agents & Multi-Tool Research Assistant

## Overview

This project demonstrates the development of a ReAct-based AI Research Assistant using LangChain, Groq, and Chainlit.

The assistant can reason about user requests, select appropriate tools, retrieve information from documents, perform calculations, access web information, and maintain conversation context.

---

## Features

### AI Agent

- ReAct-based reasoning
- Multi-tool decision making
- Autonomous tool selection

### Custom Tools

- Calculator Tool
- Web Search Tool
- RAG Search Tool
- Current Date & Time Tool

### Retrieval-Augmented Generation (RAG)

- FAISS Vector Store
- HuggingFace Embeddings
- Semantic Similarity Search

### Conversation Memory

- Maintains chat history
- Supports contextual conversations

### Performance Tracking

- Query count
- Success rate
- Average response time
- Tool usage frequency

### User Interface

- Built with Chainlit
- Interactive chatbot experience

---

## Technologies Used

- Python
- LangChain v1
- LangChain Core
- LangChain Groq
- HuggingFace Embeddings
- FAISS
- Chainlit
- Groq API
- Llama 3.3 70B Versatile

---

## Installation

```bash
pip install langchain
pip install langchain-core
pip install langchain-groq
pip install langchain-huggingface
pip install langchain-community
pip install faiss-cpu
pip install chainlit
pip install python-dotenv
```

---

## Running the Application

Start the Chainlit application:

```bash
chainlit run react_agent.py
```

---

## Performance Metrics

The application tracks:

- Total Queries
- Successful Queries
- Failed Queries
- Success Rate
- Average Response Time
- Tool Usage Frequency

---

## Learning Outcomes

- Understanding AI Agents and the ReAct framework
- Building custom LangChain tools
- Creating Retrieval-Augmented Generation systems
- Implementing conversational memory
- Tracking agent performance metrics
- Developing multi-tool AI assistants

---

## Author

Hafiza Mehak Arif

BS Artificial Intelligence | AI Engineer Intern at Xeven Solutions
