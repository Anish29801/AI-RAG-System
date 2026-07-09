# 06 — API & Backend

> **FastAPI backend design — endpoints, async processing, error handling, streaming**

---

## 1. Technology Stack

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Framework** | FastAPI | Async-native, automatic OpenAPI docs, Pydantic validation, high throughput |
| **Python** | 3.11+ | Better async support, faster CPython |
| **Server** | Uvicorn | ASGI server built for FastAPI; `--reload` for dev |
| **Validation** | Pydantic v2 | FastAPI-native; dataclass-like with serialization |
| **Async DB** | SQLAlchemy 2.0 + aiosqlite | Fully async SQLite access |
| **File handling** | python-multipart | Required for file uploads |
| **CORS** | fastapi.middleware.cors | For web UI access |

---

## 2. Application Entry Point

```python
# backend/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import AppConfig
from backend.routers import documents, chat, admin
from backend.pds.repository import PDSRepository
from backend.vector_store.chroma_client import ChromaStore
from backend.core.llm_client import OllamaClient

# Global app state (injectable via dependency)
class AppState:
    def __init__(self):
        self.config: AppConfig | None = None
        self.pds: PDSRepository | None = None
        self.vector_store: ChromaStore | None = None
        self.llm_client: OllamaClient | None = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — setup and teardown."""
    # Startup
    config = AppConfig()
    app_state.config = config
    
    app_state.pds = PDSRepository(db_path=config.pds_db_path)
    await app_state.pds.initialize()
    
    app_state.vector_store = ChromaStore(
        persist_directory=config.chroma_persist_path,
        embedding_model=config.embedding_model,
    )
    
    app_state.llm_client = OllamaClient(
        base_url=config.ollama_url,
        model=config.llm_model,
        temperature=config.temperature,
    )
    
    yield
    
    # Shutdown
    await app_state.pds.close()
    await app_state.llm_client.close()

app = FastAPI(
    title="AI RAG System",
    version="1.0.0",
    description="Free, local RAG system with LLM + Vector DB + PDS",
    lifespan=lifespan,
)

# CORS — allow web UI from any origin in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])


@app.get("/api/health")
async def health_check():
    """Basic health check."""
    llm_ok = await app_state.llm_client.is_available()
    return {
        "status": "ok",
        "llm_available": llm_ok,
        "vector_store": app_state.vector_store.get_collection_stats(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Hot reload for development
    )
```

---

## 3. Configuration

```python
# backend/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class AppConfig(BaseSettings):
    """Application configuration with env var overrides."""
    
    # LLM
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    temperature: float = 0.1
    
    # Vector Store
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_path: str = "./data/chroma_db"
    chroma_collection: str = "rag_documents"
    
    # PDS
    pds_db_path: str = "./data/pds.db"
    documents_path: str = "./data/documents"
    
    # RAG Pipeline
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    n_results: int = 10
    use_reranker: bool = False
    max_context_tokens: int = 5000
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Limits
    max_file_size_mb: int = 50
    max_concurrent_llm: int = 1
    
    class Config:
        env_file = ".env"
        env_prefix = "RAG_"
```

---

## 4. Router: Documents

```python
# backend/routers/documents.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
import os

from backend.main import app_state
from backend.core.chunker import DocumentChunker
from backend.vector_store.chroma_client import ChromaStore
from backend.pds.repository import PDSRepository
from backend.pds.file_store import FileStore

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("general"),
    tags: str = Form(""),
    description: str = Form(""),
):
    """
    Upload a document for indexing.
    
    Steps:
    1. Validate file type and size
    2. Store file on disk
    3. Register in PDS
    4. Chunk → Embed → Store in ChromaDB
    5. Return document info
    """
    # Validate
    config = app_state.config
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {config.allowed_file_types}"
        )
    
    # Read content
    content = await file.read()
    if len(content) > config.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {config.max_file_size_mb}MB limit"
        )
    
    # Store on disk
    file_store = FileStore(config.documents_path)
    file_path = file_store.store_upload(content, file.filename)
    
    # Register in PDS
    try:
        doc = await app_state.pds.add_document(
            file_path=file_path,
            filename=file.filename,
            file_type=ext.lstrip("."),
            category=category,
            tags=tags,
            description=description,
        )
    except ValueError as e:
        file_store.delete_file(file_path)
        raise HTTPException(status_code=409, detail=str(e))
    
    # Read text content
    text_content = content.decode("utf-8", errors="replace")
    
    # Chunk
    chunker = DocumentChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks, metadatas = chunker.chunk(text_content, source=file.filename)
    
    # Add metadata
    for m in metadatas:
        m["uploaded_at"] = doc.uploaded_at.isoformat()
        m["category"] = category
        m["tags"] = tags
    
    # Store embeddings
    chunk_ids = app_state.vector_store.add_document_chunks(chunks, metadatas)
    
    # Track ingestion
    ingestion = await app_state.pds.create_ingestion(doc.id, "recursive")
    await app_state.pds.update_ingestion_status(
        ingestion.id, "success",
        chunk_count=len(chunks)
    )
    
    # Update char count in document
    doc.char_count = len(text_content)
    
    return {
        "document_id": doc.id,
        "filename": file.filename,
        "chunks": len(chunks),
        "characters": len(text_content),
        "category": category,
    }


@router.get("/")
async def list_documents(
    category: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List all documents in the PDS."""
    docs = await app_state.pds.get_all_documents(
        category=category,
        file_type=file_type,
        limit=limit,
        offset=offset,
    )
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size_bytes": d.file_size_bytes,
            "category": d.category,
            "tags": d.tags.split(",") if d.tags else [],
            "uploaded_at": d.uploaded_at.isoformat(),
            "chunk_count": len(d.chunks) if d.chunks else 0,
        }
        for d in docs
    ]


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its vectors."""
    doc = await app_state.pds.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from vector store
    app_state.vector_store.delete_document(doc.filename)
    
    # Delete from PDS (cascades to file)
    await app_state.pds.delete_document(doc_id)
    
    return {"status": "deleted", "document_id": doc_id}


@router.get("/stats")
async def document_stats():
    """Get document storage statistics."""
    stats = await app_state.pds.get_document_stats()
    vector_stats = app_state.vector_store.get_collection_stats()
    return {**stats, **vector_stats}
```

---

## 5. Router: Chat

```python
# backend/routers/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
from sse_starlette.sse import EventSourceResponse
import json

from backend.main import app_state
from backend.core.rag_pipeline import RAGPipeline
from backend.core.reranker import Reranker

router = APIRouter()


# ── Request/Response Models ──

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    document_filter: Optional[str] = None  # Search within specific doc
    stream: bool = False
    use_reranker: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    latency_ms: int


class SourceInfo(BaseModel):
    source: str
    page: Optional[str] = None
    content_preview: str
    score: float


# ── Endpoints ──

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """
    Ask a question using the RAG pipeline.
    
    Non-streaming endpoint — returns complete answer.
    """
    config = app_state.config
    
    # Create or get session
    if request.session_id:
        session = await app_state.pds.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = await app_state.pds.create_session(
            title=request.query[:100]
        )
    
    # Build RAG pipeline
    reranker = Reranker() if request.use_reranker and config.use_reranker else None
    
    pipeline = RAGPipeline(
        vector_store=app_state.vector_store,
        llm_client=app_state.llm_client,
        reranker=reranker,
    )
    
    # Document filter
    where = None
    if request.document_filter:
        where = {"source": request.document_filter}
    
    # Execute
    result = await pipeline.answer(
        query=request.query,
        n_results=config.n_results,
        top_k=config.top_k,
        where=where,
        stream=False,
    )
    
    # Save to history
    await app_state.pds.add_message(
        session_id=session.id,
        role="user",
        content=request.query,
    )
    await app_state.pds.add_message(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        sources=result["sources"],
        tokens=0,
        latency=result["latency_ms"],
    )
    
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        session_id=session.id,
        latency_ms=result["latency_ms"],
    )


@router.post("/ask/stream")
async def ask_question_stream(request: ChatRequest):
    """
    Ask a question with streaming response.
    
    Uses Server-Sent Events (SSE) for streaming.
    """
    if not request.session_id:
        session = await app_state.pds.create_session(
            title=request.query[:100]
        )
    else:
        session_obj = await app_state.pds.get_session(request.session_id)
        if not session_obj:
            raise HTTPException(status_code=404, detail="Session not found")
        session = session_obj
    
    pipeline = RAGPipeline(
        vector_store=app_state.vector_store,
        llm_client=app_state.llm_client,
    )
    
    where = None
    if request.document_filter:
        where = {"source": request.document_filter}
    
    async def event_generator():
        """Generate SSE events."""
        # 1. Metadata event with sources
        result = await pipeline.answer(
            query=request.query,
            top_k=app_state.config.top_k,
            where=where,
            stream=True,
        )
        
        # Send sources first
        yield {
            "event": "sources",
            "data": json.dumps([
                {
                    "source": s["source"],
                    "page": s.get("page", ""),
                    "content_preview": s["content_preview"][:150],
                    "score": round(s["score"], 4),
                }
                for s in result["sources"]
            ])
        }
        
        # Stream tokens
        full_answer = []
        async for token in result["answer"]:
            full_answer.append(token)
            yield {
                "event": "token",
                "data": json.dumps({"token": token})
            }
        
        # Final event
        answer = "".join(full_answer)
        yield {
            "event": "done",
            "data": json.dumps({
                "latency_ms": result.get("latency_ms", 0),
                "session_id": session.id,
            })
        }
        
        # Save to history
        await app_state.pds.add_message(
            session_id=session.id,
            role="user",
            content=request.query,
        )
        await app_state.pds.add_message(
            session_id=session.id,
            role="assistant",
            content=answer,
            sources=result["sources"],
            latency=result.get("latency_ms", 0),
        )
    
    return EventSourceResponse(event_generator())


# ── Session Management ──

@router.post("/sessions")
async def create_session(title: str = "New Chat"):
    """Create a new chat session."""
    session = await app_state.pds.create_session(title=title)
    return {"session_id": session.id, "title": session.title}


@router.get("/sessions")
async def list_sessions(limit: int = 20):
    """List recent chat sessions."""
    sessions = await app_state.pds.get_recent_sessions(limit=limit)
    return [
        {
            "id": s.id,
            "title": s.title,
            "message_count": len(s.messages),
            "created_at": s.created_at.isoformat(),
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get all messages in a session."""
    messages = await app_state.pds.get_session_messages(session_id)
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "sources": m.sources_json,
            "tokens_used": m.tokens_used,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]
```

---

## 6. Router: Admin / Monitoring

```python
# backend/routers/admin.py
from fastapi import APIRouter
import psutil
import os

from backend.main import app_state

router = APIRouter()


@router.get("/health")
async def full_health_check():
    """Detailed system health check."""
    llm_ok = await app_state.llm_client.is_available()
    vector_stats = app_state.vector_store.get_collection_stats()
    pds_stats = await app_state.pds.get_document_stats()
    
    return {
        "status": "healthy" if llm_ok else "degraded",
        "components": {
            "llm": {
                "available": llm_ok,
                "model": app_state.llm_client.model,
            },
            "vector_store": vector_stats,
            "pds": pds_stats,
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_free_gb": round(
                psutil.disk_usage("/").free / (1024**3), 1
            ),
        },
    }


@router.post("/reindex")
async def reindex_all():
    """Re-index all documents (after config change)."""
    docs = await app_state.pds.get_all_documents()
    results = []
    
    for doc in docs:
        try:
            # Re-chunk and re-embed
            content = open(doc.file_path, "r", encoding="utf-8").read()
            # ... chunk, embed, store ...
            results.append({
                "filename": doc.filename,
                "status": "reindexed",
            })
        except Exception as e:
            results.append({
                "filename": doc.filename,
                "status": "failed",
                "error": str(e),
            })
    
    return {"results": results}


@router.get("/logs")
async def get_logs(lines: int = 50):
    """Return recent application logs."""
    log_file = "app.log"
    if not os.path.exists(log_file):
        return {"logs": []}
    
    with open(log_file, "r") as f:
        all_lines = f.readlines()
    
    return {"logs": all_lines[-lines:]}
```

---

## 7. Error Handling

```python
# backend/error_handlers.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("rag_api")


async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions."""
    logger.error(
        f"Unhandled error: {exc}",
        exc_info=True,
        extra={"path": request.url.path}
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred.",
            "error_type": type(exc).__name__,
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle known HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# Register in main.py:
# app.add_exception_handler(Exception, global_exception_handler)
# app.add_exception_handler(HTTPException, http_exception_handler)
```

---

## 8. Dependency Injection

```python
# backend/dependencies.py
from fastapi import Depends, HTTPException
from backend.main import app_state
from backend.pds.repository import PDSRepository
from backend.core.llm_client import OllamaClient
from backend.vector_store.chroma_client import ChromaStore


async def get_pds() -> PDSRepository:
    """Dependency: get PDS repository."""
    return app_state.pds


async def get_llm() -> OllamaClient:
    """Dependency: get LLM client."""
    return app_state.llm_client

async def get_vector_store() -> ChromaStore:
    """Dependency: get vector store."""
    return app_state.vector_store


# Usage in router:
# @router.get("/")
# async def list_docs(pds: PDSRepository = Depends(get_pds)):
#     ...
```

---

## 9. Running the Server

### Development

```bash
# From project root
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# With multiple workers (note: Ollama is single-threaded, use 1 worker)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

# Or with gunicorn (for process management)
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 10. API Documentation

With FastAPI, you get automatic OpenAPI docs:

```
Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
OpenAPI JSON: http://localhost:8000/openapi.json
```

---

## 11. Requirements

```txt
# backend/requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
sqlalchemy[asyncio]==2.0.35
aiosqlite==0.20.0
pydantic==2.9.0
pydantic-settings==2.5.0
python-multipart==0.0.9
httpx==0.27.0
sse-starlette==2.1.0
sentence-transformers==3.0.0
chromadb==0.5.0
psutil==6.0.0
python-docx==1.1.2
PyMuPDF==1.24.0
```

---

## 12. API Checklist

- [ ] FastAPI app with lifespan events initialized
- [ ] Health check endpoint working
- [ ] Document upload → chunk → embed → store flow tested end-to-end
- [ ] Chat `POST /ask` returns correct answer with sources
- [ ] Streaming `POST /ask/stream` works via SSE
- [ ] Session management (create, list, get messages) functional
- [ ] CORS configured for web UI
- [ ] Error handling catches all exceptions gracefully
- [ ] File size/type validation enforced
- [ ] Configuration loads from environment variables
- [ ] Logging configured (file + console)
- [ ] Admin endpoints report accurate system stats
