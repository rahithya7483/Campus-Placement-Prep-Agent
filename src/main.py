"""
src/main.py

FastAPI wrapper around the LangGraph agent - Module 3's Shipping Sprint.
Run with: uvicorn src.main:app --reload --port 8000
"""

import logging
import uuid
from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent import build_agent

logger = logging.getLogger("placement_agent")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] request_id=%(request_id)s %(message)s"
))
logger.addHandler(_handler)
logger.propagate = False  # don't also send these through the root logger

app = FastAPI(title="Campus Placement Prep Agent")
agent = build_agent()  # built once at startup, reused across requests


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # omit on first call; reuse the returned one for follow-ups


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    intent: str | None
    response: str


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    request_id = request.state.request_id
    session_id = req.session_id or str(uuid.uuid4())
    extra = {"request_id": request_id}

    logger.info(f"incoming message: {req.message!r} (session={session_id})", extra=extra)

    # Only pass the NEW message as input - MemorySaver restores everything
    # else (including conversation history) for this session_id automatically.
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke({"user_message": req.message}, config=config)

    if result.get("error"):
        logger.warning(f"agent reported error: {result['error']}", extra=extra)

    logger.info(f"resolved intent: {result['intent']}", extra=extra)

    return ChatResponse(
        request_id=request_id,
        session_id=session_id,
        intent=result["intent"],
        response=result["final_response"],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}