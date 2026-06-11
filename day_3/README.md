# Day 3- LLM Fundamentals & OpenAI API Setup

## Overview

Today, I focused on covering the theoretical concepts from Days 9–15 in an accelerated manner so that I could start working with LLMs and move towards LangChain, following my mentor's advice to avoid spending too much time revising topics I was already familiar with.

## Topics Covered

### Python Revision

- Tuples and tuple packing/unpacking
- Sets and set operations
- For loops and the `range()` function
- `enumerate()` and `zip()`
- `break` and `continue`
- Lambda functions
- Basics of JSON

### LLM Fundamentals

- What are Large Language Models (LLMs)
- Transformers and attention mechanisms
- Tokens and context windows
- Temperature and other common API parameters
- Foundation models
- Hallucinations and limitations of LLMs

## Practical Work Completed

- Worked on selected practical tasks from Days 9–11.
- Created an OpenAI Platform account.
- Generated an API key.
- Created and configured a `.env` file.
- Updated `.gitignore` to prevent secrets from being committed.
- Installed the required libraries:
  - `openai`
  - `python-dotenv`

- Wrote and executed my first OpenAI API call.

## Challenge Encountered

The API request reached OpenAI successfully, but the response returned the following error:

```text
Error code: 429
insufficient_quota
```

This happened because the account did not have available API credits.

## What I Learned

Even when the code is correct, external factors such as API access and service quotas can affect the outcome. I also learned the importance of securing API keys using environment variables and `.gitignore`.

## Next Steps

- Explore alternatives for completing the chatbot implementation.
- Continue with LangChain and practical LLM development.
- Build more hands-on projects using the concepts learned.
