from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import AGENT_TOOLS


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
)

llm_with_tools = llm.bind_tools(AGENT_TOOLS)


# =========================
# State
# =========================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int


# =========================
# Prompt
# =========================

SYSTEM_PROMPT = """
You are a helpful, friendly, professional AI assistant.

Follow these rules carefully.

GENERAL RULES
- Have natural conversations.
- Answer greetings naturally.
- Answer goodbyes naturally.
- Answer casual conversation yourself.
- Never use tools for greetings.
- Never use tools for goodbyes.
- Never use tools for normal conversation.
- Never mention internal reasoning.
- Never mention tools.
- Never mention function calling.
- Never expose JSON.
- Never expose tool_calls.
- Never expose raw tool output.

TOOL USAGE

Use get_weather ONLY when the user asks about:
- weather
- temperature
- forecast
- rain
- climate
- humidity
- wind

Use web_search ONLY when the answer requires:
- latest news
- current events
- recent information
- live information
- today's information
- information after your knowledge cutoff
- internet lookup

If you already know the answer,
DO NOT use web_search.

Always avoid unnecessary tool usage.

After receiving tool results,
rewrite them naturally for the user.

Never copy raw tool responses.
Never reveal tool internals.
"""


# =========================
# Nodes
# =========================

def assistant_node(state: AgentState):
    messages = state["messages"]

    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + messages
    )

    return {
        "messages": [response],
        "loop_count": state.get("loop_count", 0),
    }


tool_node = ToolNode(AGENT_TOOLS)


# =========================
# Routing
# =========================

MAX_TOOL_LOOPS = 3


def should_continue(state: AgentState):
    loop_count = state.get("loop_count", 0)

    if loop_count >= MAX_TOOL_LOOPS:
        return END

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"

    return END


def increment_loop(state: AgentState):
    return {
        "loop_count": state.get("loop_count", 0) + 1
    }


# =========================
# Graph
# =========================

graph = StateGraph(AgentState)

graph.add_node("assistant", assistant_node)
graph.add_node("tools", tool_node)
graph.add_node("increment_loop", increment_loop)

graph.add_edge(START, "assistant")

graph.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

graph.add_edge("tools", "increment_loop")
graph.add_edge("increment_loop", "assistant")

agent_graph = graph.compile()