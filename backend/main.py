"""FastAPI entry point for the AI RAG System.

Run with:  uvicorn backend.main:app --reload

Lifespan initialises PDS, ChromaDB, and LLM client on startup
and injects them into the routers via init_deps().
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import settings
from backend.pds.repository import PDSRepository
from backend.vector_store.chroma_client import ChromaStore
from backend.core.llm_client import LLMClient
from backend.routers import documents, chat, admin


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    pds = PDSRepository(db_path=settings.pds_db_path)
    await pds.initialize()

    vector = ChromaStore(
        persist_directory=settings.chroma_persist_path,
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
    )

    llm = LLMClient(
        base_url=settings.ollama_url,
        model=settings.llm_model,
        temperature=settings.temperature,
    )

    # Inject dependencies into routers
    documents.init_deps(pds, vector)
    chat.init_deps(pds, llm, vector)
    admin.init_deps(pds, llm, vector)

    yield

    # Shutdown
    await pds.close()
    await llm.close()


# ── App ──

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app = FastAPI(
    title="AI RAG System",
    version="1.0.0",
    description="Free, local RAG system with LLM + Vector DB + PDS",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])


@app.get("/api/health")
async def root_health():
    return {"status": "running", "app": "AI RAG System"}


# ── Frontend (production build) ──

dist_dir = os.path.join(frontend_dir, "dist")
if os.path.isdir(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="frontend_assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(404)
        return FileResponse(os.path.join(dist_dir, "index.html"))


# ── Runner ──

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
