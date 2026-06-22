# LangChain Agents & Tool Calling

## Agent Fundamentals

### What are AI Agents?

AI Agents are Large Language Models (LLMs) capable of making decisions and interacting with external tools to accomplish tasks autonomously. Unlike traditional LLMs that only generate text, agents can perform actions such as searching the web, querying databases, accessing documents, and performing calculations.

### ReAct Framework

The ReAct (Reason + Act) framework enables agents to solve problems through an iterative process:

1. Reason – Analyze the problem and determine what information is needed.
2. Act – Select and execute an appropriate tool.
3. Observe – Evaluate the tool's output.
4. Repeat – Continue reasoning and acting until the task is completed.

### Agent Types

- Zero-Shot Agents: Select tools without prior examples.
- Conversational Agents: Maintain conversation context across multiple interactions.
- Structured Tool Agents: Use tools with defined schemas and parameters.
- Function Calling Agents: Utilize LLM-supported function calling capabilities.

### Agent Workflow

Typical agent execution flow:

User Query
→ Planning
→ Tool Selection
→ Tool Execution
→ Result Integration
→ Final Response

### Agent Challenges

- Incorrect tool selection
- Infinite reasoning loops
- Increased token consumption
- Latency and cost management
- Error handling and recovery

---

## Retrieval Strategies

### Dense Retrieval

Uses vector embeddings and semantic similarity search to retrieve relevant information.

### Sparse Retrieval (BM25)

Uses keyword matching and statistical scoring techniques to retrieve documents.

### Hybrid Retrieval

Combines dense and sparse retrieval approaches to improve retrieval accuracy.

### Re-Ranking

Reorders retrieved documents using an additional model to improve result quality.

### Query Expansion

Enhances user queries with related terms to improve retrieval coverage.

### Self-Query Retrieval

Allows the LLM to transform user questions into structured search queries.

### Parent Document Retrieval

Retrieves larger parent documents after identifying relevant chunks to preserve context.

---

## Custom Tools Development

Implemented four LangChain tools using the `@tool` decorator:

### Calculator Tool

- Accepts mathematical expressions
- Evaluates calculations
- Includes exception handling

### Web Search Tool

- Accepts search queries
- Returns summarized search results
- Includes error handling

### RAG Search Tool

- Uses FAISS vector database
- Performs similarity search on indexed documents
- Returns relevant document chunks

### Date & Time Tool

- Returns current date and time
- Used only when explicitly requested by the user

---

## ReAct Agent Development

Built a ReAct-based AI Assistant using:

- LangChain v1
- Groq API
- Llama 3.3 70B Versatile
- Chainlit UI

### Features Implemented

- Multi-tool reasoning
- Tool selection and execution
- Document retrieval through RAG
- Mathematical calculations
- Web search capabilities
- Date and time retrieval

---

## Conversation Memory

Implemented chat history tracking to allow the agent to:

- Remember previous messages
- Maintain conversational context
- Reference earlier responses

---

## Agent Performance Tracking

Implemented:

### Query Metrics

- Total Queries
- Successful Queries
- Failed Queries

### Performance Metrics

- Success Rate
- Average Response Time

### Tool Usage Tracking

- Calculator Tool Usage
- Web Search Tool Usage
- RAG Tool Usage
- Date & Time Tool Usage

---

## Key Takeaways

- Agents extend LLM capabilities through tool usage.
- ReAct enables reasoning-driven tool selection.
- Memory significantly improves conversational experiences.
- Retrieval strategies directly impact RAG quality.
- Performance monitoring is essential for evaluating agent behavior.
- LangChain v1 simplifies agent creation and tool integration.
