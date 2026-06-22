import os

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from calculator_tool import calculator
from web_search_tool import web_search
from rag_tool import rag_search

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[2] / ".env"

print("Looking for:", env_path)
print("Exists:", env_path.exists())

load_dotenv(env_path)

print("API Key:", os.getenv("GROQ_API_KEY"))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

tools = [
    calculator,
    web_search,
    rag_search
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are a helpful AI assistant.

    Use tools whenever necessary.
    For calculations use calculator.
    For internet information use web_search.
    For document-related questions use rag_search.

    Always provide a final answer.
    """
)

while True:

    query = input("\nYou: ")

    if query.lower() == "exit":
        break

    try:

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )

        print("\nAgent:")

        print(
            response["messages"][-1].content
        )

    except Exception as e:

        print(f"\nError: {e}")