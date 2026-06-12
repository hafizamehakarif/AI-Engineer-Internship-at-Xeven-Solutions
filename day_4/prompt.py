from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

formatted_prompt = prompt.invoke({"topic": "Transformers"})

print(formatted_prompt)