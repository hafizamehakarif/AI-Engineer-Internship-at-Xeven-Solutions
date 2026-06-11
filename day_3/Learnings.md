# Learnings.md

## What I Learned Today

Today, I focused mainly on covering the theoretical concepts from Days 9–15 so that I can move towards working with LLMs and LangChain sooner.

### Python Concepts

- Tuples and why they are immutable.
- Tuple packing and unpacking.
- Sets and how they store only unique values.
- Basic set operations such as union, intersection, and difference.
- For loops and the `range()` function.
- Using `enumerate()` and `zip()`.
- The use of `break` and `continue` in loops.
- Lambda functions.
- Basic JSON structure and why it is commonly used.

### LLM Concepts

- What Large Language Models (LLMs) are.
- Basic idea of transformers and attention mechanisms.
- Tokens and how models process text.
- Context windows and their limitations.
- Temperature and how it affects responses.
- Hallucinations and why model outputs should be verified.
- Foundation models and their role in modern AI.

## Practical Work

- Worked on some of the practical tasks from Days 9–11.
- Created an OpenAI Platform account.
- Generated an API key.
- Configured a `.env` file and updated `.gitignore` to keep secrets safe.
- Installed the required libraries for OpenAI integration.
- Wrote and executed my first OpenAI API call.

## Challenges Faced

- Faced an issue where `.gitignore` was not ignoring the `.env` file because of an encoding problem. After changing it to UTF-8, the issue was resolved.
- The OpenAI API call returned an `insufficient_quota` error because the account did not have available API credits.

## Reflection

Today's work felt like a shift from revising Python concepts to actually setting up the tools needed for LLM development. Even though the API call could not generate a response due to quota limitations, I was able to complete the setup process and understand how these integrations work in practice.
