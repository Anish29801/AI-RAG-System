"""Async CRUD repository for the Personal Data Store.

Uses SQLAlchemy 2.0 async with aiosqlite for non-blocking
database operations on the local SQLite database.
"""

import os
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker,
)
from sqlalchemy import select, func

from backend.pds.models import (
    Base, Document, DocumentChunk, IngestionRecord,
    ChatSession, ChatMessage,
)
from backend.pds.file_store import FileStore


def _utcnow():
    return datetime.now(timezone.utc)


class PDSRepository:
    """Async repository for Personal Data Store operations."""

    def __init__(self, db_path: str = "./data/pds.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False,
        )

    async def initialize(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def _session(self) -> AsyncSession:
        return self.session_factory()

    # ── Documents ──

    async def add_document(
        self, filename: str, file_path: str, file_type: str,
        file_size_bytes: int, file_hash: str,
        category: str = "general", tags: str = "", description: str = "",
    ) -> Document:
        async with self._session() as session:
            doc = Document(
                filename=filename, file_path=file_path,
                file_type=file_type, file_size_bytes=file_size_bytes,
                file_hash=file_hash, category=category,
                tags=tags, description=description,
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            return doc

    async def get_document(self, doc_id: str) -> Document | None:
        async with self._session() as session:
            return await session.get(Document, doc_id)

    async def get_document_by_hash(self, file_hash: str) -> Document | None:
        async with self._session() as session:
            result = await session.execute(
                select(Document).where(Document.file_hash == file_hash)
            )
            return result.scalar_one_or_none()

    async def get_all_documents(
        self, category: str | None = None, file_type: str | None = None,
        limit: int = 50, offset: int = 0,
    ) -> list[Document]:
        async with self._session() as session:
            query = select(Document).order_by(Document.uploaded_at.desc())
            if category:
                query = query.where(Document.category == category)
            if file_type:
                query = query.where(Document.file_type == file_type)
            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def delete_document(self, doc_id: str) -> bool:
        async with self._session() as session:
            doc = await session.get(Document, doc_id)
            if not doc:
                return False
            FileStore().delete(doc.file_path)
            await session.delete(doc)
            await session.commit()
            return True

    async def get_document_stats(self) -> dict:
        async with self._session() as session:
            total = await session.scalar(select(func.count(Document.id))) or 0
            total_chars = await session.scalar(
                select(func.coalesce(func.sum(Document.char_count), 0))
            )
            result = await session.execute(
                select(Document.file_type, func.count(Document.id))
                .group_by(Document.file_type)
            )
            by_type = dict(result.all())
            return {
                "total_documents": total,
                "total_chars": total_chars,
                "documents_by_type": by_type,
            }

    # ── Ingestion ──

    async def create_ingestion(
        self, document_id: str, strategy: str,
    ) -> IngestionRecord:
        async with self._session() as session:
            rec = IngestionRecord(
                document_id=document_id, strategy=strategy, status="running",
                started_at=_utcnow(),
            )
            session.add(rec)
            await session.commit()
            await session.refresh(rec)
            return rec

    async def update_ingestion_status(
        self, ingestion_id: str, status: str,
        chunk_count: int = 0, error_message: str = "",
    ):
        async with self._session() as session:
            rec = await session.get(IngestionRecord, ingestion_id)
            if rec:
                rec.status = status
                rec.chunk_count = chunk_count
                if error_message:
                    rec.error_message = error_message
                if status in ("success", "failed"):
                    rec.completed_at = _utcnow()
                await session.commit()

    # ── Chat Sessions ──

    async def create_session(
        self, title: str = "New Chat", model: str = "llama3.1:8b",
    ) -> ChatSession:
        async with self._session() as session:
            chat = ChatSession(title=title, model_used=model)
            session.add(chat)
            await session.commit()
            await session.refresh(chat)
            return chat

    async def get_session(self, session_id: str) -> ChatSession | None:
        async with self._session() as session:
            return await session.get(ChatSession, session_id)

    async def get_recent_sessions(self, limit: int = 20) -> list[ChatSession]:
        async with self._session() as session:
            result = await session.execute(
                select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)
            )
            return list(result.scalars().all())

    # ── Messages ──

    async def add_message(
        self, session_id: str, role: str, content: str,
        sources: list[dict] | None = None,
        tokens: int = 0, latency: int = 0,
    ) -> ChatMessage:
        async with self._session() as session:
            msg = ChatMessage(
                session_id=session_id, role=role, content=content,
                sources_json=sources, tokens_used=tokens, latency_ms=latency,
            )
            session.add(msg)
            # Touch session updated_at
            chat = await session.get(ChatSession, session_id)
            if chat:
                chat.updated_at = _utcnow()
            await session.commit()
            await session.refresh(msg)
            return msg

    async def get_session_messages(
        self, session_id: str, limit: int = 100,
    ) -> list[ChatMessage]:
        async with self._session() as session:
            result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
                .limit(limit)
            )
            return list(result.scalars().all())

    # ── Teardown ──

    async def close(self):
        await self.engine.dispose()
