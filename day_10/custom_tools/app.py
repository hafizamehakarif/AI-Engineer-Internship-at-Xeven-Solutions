import os
import time
import asyncio
import traceback
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

# Tools
from calculator_tool import calculator
from web_search_tool import web_search
from rag_tool import rag_search
from current_datetime_tool import get_current_datetime

# Load Environment Variables
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Tools List
tools = [
    calculator,
    web_search,
    rag_search,
    get_current_datetime
]

# Agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="""You are a helpful AI assistant.

Only use a tool when it is required to answer the user's question.

Available tools:

* calculator: mathematical calculations
* web_search: internet information
* rag_search: document knowledge
* get_current_datetime: current date and time

Do not call get_current_datetime unless the user explicitly asks for the current date, time, day, month, or year.

Answer directly when no tool is needed.
"""
)

# Memory
chat_history = []

# Performance Tracking
total_queries = 0
successful_queries = 0
failed_queries = 0
response_times = []

# Memory Window
MAX_TURNS = 5


@cl.on_chat_start
async def start():
    global chat_history
    chat_history = []

    await cl.Message(
        content="""# ReAct AI Assistant

Available Tools:

* Calculator
* Web Search
* RAG Search
* Current Date & Time

Memory Enabled
Performance Tracking Enabled
"""
    ).send()


@cl.on_message
async def main(message: cl.Message):
    global chat_history
    global total_queries
    global successful_queries
    global failed_queries
    global response_times

    total_queries += 1
    start_time = time.time()

    try:
        chat_history.append({"role": "user", "content": message.content})

        response = agent.invoke({"messages": chat_history})

        assistant_reply = response["messages"][-1].content

        chat_history.append({"role": "assistant", "content": assistant_reply})

        chat_history = chat_history[-(MAX_TURNS * 2):]

        successful_queries += 1

        await cl.Message(content=assistant_reply).send()

    except Exception as e:
        failed_queries += 1
        await cl.Message(content=f"Error: {e}").send()

    finally:
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        response_times.append(response_time)

        average_response_time = sum(response_times) / len(response_times)
        success_rate = (successful_queries / total_queries) * 100

        print("\n===== Agent Performance =====")
        print(f"Total Queries: {total_queries}")
        print(f"Successful Queries: {successful_queries}")
        print(f"Failed Queries: {failed_queries}")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Average Response Time: {average_response_time:.2f} sec")