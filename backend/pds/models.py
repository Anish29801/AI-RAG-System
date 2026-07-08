"""SQLAlchemy ORM models for the Personal Data Store.

Tables:
  - documents           : File registry and metadata
  - document_chunks     : Individual chunks with vector_id linkage
  - ingestion_records   : Track ingestion status per document
  - chat_sessions       : Conversation sessions
  - chat_messages       : Individual messages per session
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, DateTime, Text,
    ForeignKey, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Document ──


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(32), nullable=False)
    page_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    category = Column(String(100), default="general")
    tags = Column(String(500), default="")
    description = Column(Text, default="")
    uploaded_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    chunks = relationship(
        "DocumentChunk", back_populates="document",
        cascade="all, delete-orphan",
    )
    ingestions = relationship(
        "IngestionRecord", back_populates="document",
        cascade="all, delete-orphan",
    )


# ── DocumentChunk ──


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(
        String, ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    char_count = Column(Integer, default=0)
    vector_id = Column(String, index=True)
    metadata_json = Column(JSON, default=dict)

    document = relationship("Document", back_populates="chunks")


# ── IngestionRecord ──


class IngestionRecord(Base):
    __tablename__ = "ingestion_records"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(
        String, ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    strategy = Column(String(50), nullable=False)
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="pending")
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    document = relationship("Document", back_populates="ingestions")


# ── ChatSession ──


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=_uuid)
    title = Column(String(200), default="New Chat")
    model_used = Column(String(100), default="llama3.1:8b")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    messages = relationship(
        "ChatMessage", back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


# ── ChatMessage ──


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(
        String, ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    sources_json = Column(JSON, nullable=True)
    tokens_used = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)

    session = relationship("ChatSession", back_populates="messages")
