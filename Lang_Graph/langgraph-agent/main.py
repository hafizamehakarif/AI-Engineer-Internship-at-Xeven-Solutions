import os
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------

parent_env = Path(__file__).resolve().parent.parent / ".env"

if parent_env.exists():
    load_dotenv(dotenv_path=parent_env)
else:
    load_dotenv()

# -----------------------------------------------------------------------------
# LOCAL IMPORTS
# -----------------------------------------------------------------------------

from config import get_settings
from database.db import init_db
from database.store import (
    initialize_session,
    save_chat_message,
    get_sliding_window_history,
    delete_session,
)
from agent import agent_graph

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# -----------------------------------------------------------------------------
# FASTAPI LIFESPAN
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    yield
    logger.info("Application shutdown.")

# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="LangGraph Conversational Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# REQUEST MODELS
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class SessionCreateRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def extract_clean_string(content: Any) -> str:
    """
    Extracts only the readable text from any LangChain/Gemini response.
    """

    if hasattr(content, "content"):
        content = content.content

    # Normal string
    if isinstance(content, str):
        return content.strip()

    # Gemini content blocks
    if isinstance(content, list):
        texts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                texts.append(block.text)
            else:
                texts.append(str(block))

        return "\n".join(filter(None, texts)).strip()

    # Dict response
    if isinstance(content, dict):

        if "text" in content:
            return str(content["text"]).strip()

        if "content" in content:
            return extract_clean_string(content["content"])

        return json.dumps(content)

    return str(content).strip()

# -----------------------------------------------------------------------------
# HEALTH
# -----------------------------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {
        "status": "healthy",
        "service": "langgraph-agent-backend",
    }

# -----------------------------------------------------------------------------
# SESSION INIT
# -----------------------------------------------------------------------------

@app.post("/session/init")
def create_session(payload: SessionCreateRequest):

    try:
        result = initialize_session(
            user_id=payload.user_id,
            session_id=payload.session_id,
        )

        if result["status"] == "invalid_session":
            raise HTTPException(
                status_code=404,
                detail=result["message"],
            )

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize session.",
        )

# -----------------------------------------------------------------------------
# CHAT
# -----------------------------------------------------------------------------

@app.post("/chat")
def chat(payload: ChatRequest):

    try:

        # -------------------------------
        # Validate Session
        # -------------------------------

        session = initialize_session(
            user_id=payload.user_id,
            session_id=payload.session_id,
        )

        if session["status"] == "invalid_session":
            raise HTTPException(
                status_code=404,
                detail=session["message"],
            )

        # -------------------------------
        # Load Conversation History
        # -------------------------------

        history = get_sliding_window_history(
            payload.session_id,
            limit=5,
        )

        formatted_history = []

        for msg in history:

            if isinstance(msg, dict):

                role = msg.get("role")
                content = msg.get("content", "")

                if role == "user":
                    formatted_history.append(
                        HumanMessage(content=content)
                    )

                elif role in ("assistant", "model"):
                    formatted_history.append(
                        AIMessage(content=content)
                    )

            else:
                formatted_history.append(msg)

        # -------------------------------
        # Save User Message
        # -------------------------------

        user_message = extract_clean_string(payload.message)

        save_chat_message(
            session_id=payload.session_id,
            role="user",
            content=user_message,
        )

        # -------------------------------
        # Invoke Graph
        # -------------------------------

        state = {
            "messages": formatted_history
            + [HumanMessage(content=user_message)],
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "loop_count": 0,
        }

        output = agent_graph.invoke(state)

        # -------------------------------
        # Read Final AI Message
        # -------------------------------

        messages = output.get("messages", [])

        if not messages:
            raise HTTPException(
                status_code=500,
                detail="Graph returned no messages.",
            )

        last_message = messages[-1]

        logger.info("FINAL MESSAGE TYPE: %s", type(last_message))
        logger.info("FINAL MESSAGE: %s", last_message)

        final_response = extract_clean_string(last_message)

        # -------------------------------
        # Save Assistant Response
        # -------------------------------

        if final_response:

            save_chat_message(
                session_id=payload.session_id,
                role="assistant",
                content=final_response,
            )

        # -------------------------------
        # API Response
        # -------------------------------

        return {
            "status": "success",
            "response": final_response,
        }

    except HTTPException:
        raise

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=f"Execution error: {e}",
        )

# -----------------------------------------------------------------------------
# DELETE SESSION
# -----------------------------------------------------------------------------

@app.delete("/session/{session_id}")
def delete_chat_session(session_id: str):

    try:

        success = delete_session(session_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )

        return {
            "status": "success",
            "message": f"Session '{session_id}' deleted successfully.",
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail="Database deletion failed.",
        )