# Week 2 – Day 1 Learnings

## Day 17: Embeddings and Semantic Search

### What I Learned

- Embeddings convert text into numerical vector representations while preserving semantic meaning.
- Similar texts have embeddings that are closer together in vector space.
- Cosine similarity is commonly used to measure how similar two embeddings are.
- OpenAI embeddings are not mandatory; open-source models such as Sentence Transformers can also be used effectively.
- Semantic search retrieves information based on meaning rather than exact keyword matching.
- Traditional databases are not optimized for similarity search, which is why vector databases exist.

### Practical Takeaways

- Generated embeddings using Sentence Transformers.
- Implemented cosine similarity from scratch.
- Compared similarities between related and unrelated sentences.
- Built a simple semantic search system.
- Developed a chatbot using Groq and understood the importance of controlling conversation history to reduce token usage.
- Learned that not every problem requires an LLM; deterministic solutions are often more efficient.

---

## Day 18: Text Splitters and Chunking Strategies

### What I Learned

- Chunking is essential because LLMs and embedding models have context limitations.
- Different chunking strategies have different strengths:
  - Fixed-size chunking
  - Recursive chunking
  - Sentence-based chunking
  - Semantic chunking

- Overlap helps preserve context between chunks.
- Smaller chunks improve retrieval precision, while larger chunks preserve more context.

### Practical Takeaways

- Compared different chunking approaches conceptually.
- Explored LangChain splitters such as:
  - RecursiveCharacterTextSplitter
  - MarkdownHeaderTextSplitter
  - PythonCodeTextSplitter

- Built an interactive Smart Document Processor.
- Added document type detection and splitter recommendations.
- Preserved metadata such as headers and function names.
- Encountered real-world encoding issues and learned how to handle them.

---

## Day 19: Prompt Engineering Mastery

### What I Learned

- Zero-shot prompting works well for simple tasks.
- Few-shot prompting improves consistency by providing examples.
- Chain-of-Thought prompting is useful for reasoning-intensive problems.
- System messages define the assistant's behavior, tone, and constraints.
- Effective prompts usually include:
  - Persona
  - Instructions
  - Context
  - Format requirements
  - Examples
  - Constraints

### Common Mistakes to Avoid

- Writing vague prompts.
- Assuming the model understands unstated context.
- Forgetting to specify output formats.
- Giving conflicting instructions.
- Adding unnecessary constraints.
