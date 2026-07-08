"""Chat and RAG query endpoints.

- POST /ask                      — non-streaming Q&A with RAG
- POST /ask/stream               — streaming Q&A via SSE
- POST /sessions                 — create chat session
- GET  /sessions                 — list sessions
- GET  /sessions/{id}/messages   — get message history
"""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from sse_starlette.sse import EventSourceResponse

from backend.config import settings
from backend.pds.repository import PDSRepository
from backend.core.rag_pipeline import RAGPipeline
from backend.core.llm_client import LLMClient
from backend.vector_store.chroma_client import ChromaStore

router = APIRouter()


# ── Dependencies (set by main.py lifespan) ──

_pds: Optional[PDSRepository] = None
_llm: Optional[LLMClient] = None
_vector: Optional[ChromaStore] = None


def init_deps(pds: PDSRepository, llm: LLMClient, vector: ChromaStore):
    global _pds, _llm, _vector
    _pds = pds
    _llm = llm
    _vector = vector


# ── Request / Response Models ──


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    document_filter: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    latency_ms: int


# ── Endpoints ──


@router.post("/ask")
async def ask_question(request: ChatRequest):
    if not _pds or not _llm or not _vector:
        raise HTTPException(503, "System not initialised")

    session = None
    if request.session_id:
        session = await _pds.get_session(request.session_id)
        if not session:
            raise HTTPException(404, "Session not found")

    if not session:
        session = await _pds.create_session(title=request.query[:100])

    pipeline = RAGPipeline(vector_store=_vector, llm_client=_llm)

    where = None
    if request.document_filter:
        where = {"source": request.document_filter}

    result = await pipeline.answer(
        query=request.query,
        n_results=settings.n_results,
        top_k=settings.top_k,
        where=where,
        stream=False,
    )

    # Save messages
    await _pds.add_message(session.id, "user", request.query)
    await _pds.add_message(
        session.id, "assistant", result["answer"],
        sources=result["sources"],
        latency=result["latency_ms"],
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=session.id,
        latency_ms=result["latency_ms"],
    )


@router.post("/ask/stream")
async def ask_question_stream(request: ChatRequest):
    if not _pds or not _llm or not _vector:
        raise HTTPException(503, "System not initialised")

    session = None
    if request.session_id:
        session = await _pds.get_session(request.session_id)
        if not session:
            raise HTTPException(404, "Session not found")

    if not session:
        session = await _pds.create_session(title=request.query[:100])

    pipeline = RAGPipeline(vector_store=_vector, llm_client=_llm)

    where = None
    if request.document_filter:
        where = {"source": request.document_filter}

    result = await pipeline.answer(
        query=request.query,
        n_results=settings.n_results,
        top_k=settings.top_k,
        where=where,
        stream=True,
    )

    sources_data = [
        {
            "source": s["source"],
            "content_preview": s["content_preview"][:150],
            "score": round(s["score"], 4),
        }
        for s in result["sources"]
    ]

    async def event_generator():
        # 1. Send sources first
        yield {"event": "sources", "data": json.dumps(sources_data)}

        # 2. Stream answer tokens
        full_answer = []
        async for token in result["answer"]:
            full_answer.append(token)
            yield {"event": "token", "data": json.dumps({"token": token})}

        answer = "".join(full_answer)

        # 3. Done event
        yield {
            "event": "done",
            "data": json.dumps({
                "latency_ms": result.get("latency_ms", 0),
                "session_id": session.id,
            }),
        }

        # 4. Persist to history (fire-and-forget)
        await _pds.add_message(session.id, "user", request.query)
        await _pds.add_message(
            session.id, "assistant", answer,
            sources=result["sources"],
            latency=result.get("latency_ms", 0),
        )

    return EventSourceResponse(event_generator())


# ── Session Management ──


@router.post("/sessions")
async def create_session(title: str = "New Chat"):
    if not _pds:
        raise HTTPException(503, "System not initialised")
    session = await _pds.create_session(title=title)
    return {"session_id": session.id, "title": session.title}


@router.get("/sessions")
async def list_sessions(limit: int = 20):
    if not _pds:
        raise HTTPException(503, "System not initialised")
    sessions = await _pds.get_recent_sessions(limit=limit)
    return [
        {
            "id": s.id,
            "title": s.title,
            "model_used": s.model_used,
            "created_at": s.created_at.isoformat(),
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    if not _pds:
        raise HTTPException(503, "System not initialised")
    messages = await _pds.get_session_messages(session_id)
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "sources": m.sources_json,
            "tokens_used": m.tokens_used,
            "latency_ms": m.latency_ms,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]
