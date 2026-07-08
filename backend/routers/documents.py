"""Document management endpoints.

- POST /upload          — upload and index a document
- GET  /                — list all documents
- GET  /{doc_id}        — get a single document
- DELETE /{doc_id}      — delete document and vectors
- GET  /stats           — storage statistics
"""

import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional

from backend.config import settings
from backend.pds.repository import PDSRepository
from backend.pds.file_store import FileStore
from backend.vector_store.chroma_client import ChromaStore
from backend.core.chunker import DocumentChunker

router = APIRouter()


# ── Dependencies (set by main.py lifespan) ──

_pds: Optional[PDSRepository] = None
_vector: Optional[ChromaStore] = None


def init_deps(pds: PDSRepository, vector: ChromaStore):
    global _pds, _vector
    _pds = pds
    _vector = vector


# ── Endpoints ──


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("general"),
    tags: str = Form(""),
    description: str = Form(""),
):
    if not _pds or not _vector:
        raise HTTPException(503, "System not initialised")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.allowed_file_types:
        raise HTTPException(400, f"File type '{ext}' not allowed")

    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_file_size_mb} MB limit")

    # Store on disk
    file_store = FileStore(settings.documents_path)
    file_path = file_store.store(content, file.filename or "upload")

    # Register in PDS
    file_hash = FileStore.md5_bytes(content)
    existing = await _pds.get_document_by_hash(file_hash)
    if existing:
        file_store.delete(file_path)
        raise HTTPException(409, "Duplicate document (same content already uploaded)")

    doc = await _pds.add_document(
        filename=file.filename or "unknown",
        file_path=file_path,
        file_type=ext.lstrip("."),
        file_size_bytes=len(content),
        file_hash=file_hash,
        category=category,
        tags=tags,
        description=description,
    )

    # Read text and chunk
    text = content.decode("utf-8", errors="replace")
    chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
    chunks, metadatas = chunker.chunk(text, source=file.filename or "upload")

    for m in metadatas:
        m["category"] = category
        m["tags"] = tags

    # Store vectors
    _vector.add_chunks(chunks, metadatas)

    # Track ingestion
    ingestion = await _pds.create_ingestion(doc.id, "auto")
    await _pds.update_ingestion_status(ingestion.id, "success", chunk_count=len(chunks))

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "chunks": len(chunks),
        "characters": len(text),
        "category": category,
    }


@router.get("/")
async def list_documents(
    category: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    if not _pds:
        raise HTTPException(503, "System not initialised")
    docs = await _pds.get_all_documents(category, file_type, limit, offset)
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size_bytes": d.file_size_bytes,
            "category": d.category,
            "tags": d.tags.split(",") if d.tags else [],
            "char_count": d.char_count,
            "uploaded_at": d.uploaded_at.isoformat(),
        }
        for d in docs
    ]


@router.get("/stats")
async def document_stats():
    if not _pds or not _vector:
        raise HTTPException(503, "System not initialised")
    pds_stats = await _pds.get_document_stats()
    vector_stats = _vector.get_stats()
    return {**pds_stats, "vector_chunks": vector_stats["total_chunks"]}


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    if not _pds:
        raise HTTPException(503, "System not initialised")
    doc = await _pds.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {
        "id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "file_size_bytes": doc.file_size_bytes,
        "category": doc.category,
        "tags": doc.tags.split(",") if doc.tags else [],
        "char_count": doc.char_count,
        "description": doc.description,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    if not _pds or not _vector:
        raise HTTPException(503, "System not initialised")
    doc = await _pds.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    _vector.delete_document(doc.filename)
    await _pds.delete_document(doc_id)
    return {"status": "deleted", "document_id": doc_id}
