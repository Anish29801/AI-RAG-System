"""Admin and monitoring endpoints.

- GET  /health      — full system health check
- POST /reindex     — re-index all documents
- GET  /stats       — comprehensive system stats
"""

import os
import time
from fastapi import APIRouter, HTTPException
from typing import Optional

from backend.pds.repository import PDSRepository
from backend.core.llm_client import LLMClient
from backend.vector_store.chroma_client import ChromaStore

router = APIRouter()


# ── Dependencies (set by main.py lifespan) ──

_pds: Optional[PDSRepository] = None
_llm: Optional[LLMClient] = None
_vector: Optional[ChromaStore] = None
_start_time: float = 0.0


def init_deps(pds: PDSRepository, llm: LLMClient, vector: ChromaStore):
    global _pds, _llm, _vector, _start_time
    _pds = pds
    _llm = llm
    _vector = vector
    _start_time = time.time()


# ── Endpoints ──


@router.get("/health")
async def health_check():
    if not _llm:
        raise HTTPException(503, "System not initialised")
    llm_ok = await _llm.is_available()
    vector_stats = _vector.get_stats() if _vector else {"total_chunks": 0}
    uptime = time.time() - _start_time

    return {
        "status": "healthy" if llm_ok else "degraded",
        "uptime_seconds": round(uptime),
        "components": {
            "llm": {"available": llm_ok, "model": _llm.model},
            "vector_store": vector_stats,
        },
    }


@router.get("/stats")
async def system_stats():
    if not _pds or not _vector:
        raise HTTPException(503, "System not initialised")
    pds_stats = await _pds.get_document_stats()
    vector_stats = _vector.get_stats()
    uptime = time.time() - _start_time

    return {
        "uptime_seconds": round(uptime),
        "documents": pds_stats,
        "vectors": vector_stats,
        "memory": _get_memory_info(),
    }


@router.post("/reindex")
async def reindex_all():
    """Re-chunk and re-embed all documents."""
    if not _pds or not _vector:
        raise HTTPException(503, "System not initialised")

    docs = await _pds.get_all_documents(limit=9999)
    results = []

    for doc in docs:
        try:
            # Read from disk
            if not os.path.exists(doc.file_path):
                results.append({"filename": doc.filename, "status": "file_missing"})
                continue

            with open(doc.file_path, "rb") as f:
                content = f.read().decode("utf-8", errors="replace")

            # Delete old vectors
            _vector.delete_document(doc.filename)

            # Re-chunk and re-embed
            from backend.core.chunker import DocumentChunker
            from backend.config import settings
            chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
            chunks, metadatas = chunker.chunk(content, source=doc.filename)
            _vector.add_chunks(chunks, metadatas)

            results.append({
                "filename": doc.filename,
                "status": "reindexed",
                "chunks": len(chunks),
            })
        except Exception as e:
            results.append({
                "filename": doc.filename,
                "status": "failed",
                "error": str(e),
            })

    return {"results": results}


def _get_memory_info() -> dict:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_mb": round(mem.total / (1024**2)),
            "available_mb": round(mem.available / (1024**2)),
            "percent_used": mem.percent,
        }
    except ImportError:
        return {}
