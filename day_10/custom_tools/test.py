"""
Quick test to check which Groq models support tool calling reliably.
Run this BEFORE starting Chainlit: python test_tool_calling.py
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# .env file ka hardcoded path — root folder mein hai
env_path = Path("C:/Users/PMLS/Desktop/AI_Internship_Xeven_2026/.env")
load_dotenv(env_path)

# Verify key load hui ya nahi
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY nahi mili — .env file check karein")
    print(f"   Path check kiya: {env_path}")
    exit(1)
else:
    print(f"✅ GROQ_API_KEY mili — (ends with: ...{api_key[-6:]})\n")

from langchain_groq import ChatGroq
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """
    Search the internet for current news and information.

    Args:
        query: The search query string to look up on the web.

    Returns:
        A string containing relevant search results.
    """
    return f"Results for: {query}"


MODELS_TO_TEST = [
    "llama-3.3-70b-versatile",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama-3.1-70b-versatile",
]

print("Testing tool calling on Groq models...\n")

for model_name in MODELS_TO_TEST:
    try:
        llm = ChatGroq(
            model=model_name,
            temperature=0,
            api_key=api_key
        )
        llm_with_tools = llm.bind_tools([web_search])
        response = llm_with_tools.invoke("What is the latest news today?")
        called = bool(getattr(response, "tool_calls", []))
        print(f"{'✅' if called else '❌'} {model_name}")
        if called:
            print(f"   Tool called: {response.tool_calls[0]['name']}")
            print(f"   Args: {response.tool_calls[0]['args']}")
        else:
            print(f"   No tool called — model answered directly instead")
    except Exception as e:
        print(f"💥 {model_name} — Error: {e}")
    print()