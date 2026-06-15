# Day 19 – Prompt Engineering Mastery

## What I Learned

### Zero-Shot Prompting

Giving the model only instructions without examples.

**Best for:**

- Summarization
- Translation
- Simple classification

**Pros:**

- Fast
- Cheap

**Cons:**

- Less reliable on complex tasks

---

### Few-Shot Prompting

Providing a few examples to guide the model.

**Best for:**

- Structured outputs
- Information extraction
- Domain-specific tasks

**Pros:**

- Better accuracy
- More consistent formatting

**Cons:**

- Uses more tokens

---

### Chain-of-Thought Prompting

Encouraging the model to reason step by step.

**Best for:**

- Logic problems
- Planning
- Mathematical reasoning

**Pros:**

- Improves reasoning quality

**Cons:**

- Longer responses
- Higher token usage

---

### System Messages

Used to define the assistant's role, tone, constraints, and output style.

Example:
"You are a senior Python tutor. Explain concepts simply and keep responses concise."

---

## Common Prompting Mistakes

- Giving vague instructions
- Assuming the model knows the context
- Not specifying the desired output format
- Using conflicting instructions
- Adding unnecessary constraints

---

## My Biggest Takeaway

There is no single "best" prompt. The right prompting technique depends on the task:

- Use zero-shot for simple tasks.
- Use few-shot when consistency matters.
- Use Chain-of-Thought for reasoning problems.
- Refine prompts through experimentation rather than expecting perfection on the first try.
