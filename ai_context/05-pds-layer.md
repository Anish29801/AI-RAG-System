# 05 — Personal Data Store (PDS)

> **Document management, metadata tracking, versioning, file storage — all free, local, and private**

---

## 1. What is the PDS Layer?

The Personal Data Store is the **document registry** and **metadata backbone** of the system. While ChromaDB stores vector embeddings, the PDS tracks:

- **Files**: Where they live on disk, their names, sizes, hashes
- **Metadata**: Source, upload date, category, tags, file type
- **Ingestion State**: Which files have been indexed, chunked, synced
- **Conversations**: Chat history, queries, retrieval traces
- **Configuration**: User preferences, system settings

### Layer Interaction

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  File System │────►│     PDS      │────►│   ChromaDB   │
│  (documents) │     │   (SQLite)   │     │  (vectors)   │
│              │     │              │     │              │
│  raw files   │     │  metadata    │     │  embeddings  │
│  on disk     │     │  registry    │     │  + chunks    │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 2. Technology: SQLite

### Why SQLite for PDS

| Requirement | SQLite | PostgreSQL | MongoDB |
|-------------|--------|-----------|---------|
| **Free** | ✅ Public Domain | ✅ Open source | ✅ Open source |
| **Zero config** | ✅ No server | ❌ Needs server | ❌ Needs config |
| **Portable** | ✅ Single file | ❌ Data dir | ❌ Data dir |
| **Atomic writes** | ✅ Full ACID | ✅ Full ACID | ❌ Doc-level |
| **Performance** | ★★★★★ | ★★★★ | ★★★★ |
| **Backup** | ✅ One file copy | ✅ pg_dump | ✅ mongodump |
| **Async** | ✅ aiosqlite | ✅ asyncpg | ✅ motor |

### Installation

```bash
# SQLite is built into Python 3.11+
# For async support:
pip install aiosqlite
# For ORM:
pip install sqlalchemy[asyncio] aiosqlite
```

---

## 3. Database Schema

### Entity-Relationship Diagram

```
┌──────────────────┐       ┌──────────────────────┐
│     Document     │       │   IngestionRecord    │
├──────────────────┤       ├──────────────────────┤
│ id (PK)          │──┐   │ id (PK)              │
│ filename         │  └───►│ document_id (FK)     │
│ file_path        │       │ strategy             │
│ file_type        │       │ chunk_count          │
│ file_size_bytes  │       │ status               │
│ file_hash (md5)  │       │ error_message        │
│ page_count       │       │ started_at           │
│ char_count       │       │ completed_at         │
│ category         │       └──────────────────────┘
│ tags             │
│ uploaded_at      │       ┌──────────────────────┐
│ updated_at       │       │      ChatSession     │
│ description      │       ├──────────────────────┤
└──────────────────┘       │ id (PK)              │
                           │ title                │
┌──────────────────┐       │ model_used           │
│  DocumentChunk   │       │ created_at           │
├──────────────────┤       │ updated_at           │
│ id (PK)          │       └──────────────────────┘
│ document_id (FK) │              │
│ chunk_index      │              ▼
│ content          │       ┌──────────────────────┐
│ char_count       │       │     ChatMessage      │
│ vector_id        │       ├──────────────────────┤
│ metadata_json    │       │ id (PK)              │
└──────────────────┘       │ session_id (FK)      │
                           │ role (user/assistant)│
                           │ content              │
                           │ sources_json         │
                           │ tokens_used          │
                           │ created_at           │
                           └──────────────────────┘
```

### SQLAlchemy Models

```python
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text,
    ForeignKey, JSON, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, timezone
import uuid

class Base(DeclarativeBase):
    pass

def _utcnow():
    return datetime.now(timezone.utc)

def _uuid():
    return str(uuid.uuid4())

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False)
    file_type = Column(String(10), nullable=False)  # pdf, txt, md, docx
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(32), nullable=False)  # MD5 for dedup
    page_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    category = Column(String(100), default="general")
    tags = Column(String(500), default="")  # comma-separated
    description = Column(Text, default="")
    uploaded_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    chunks = relationship("DocumentChunk", back_populates="document",
                          cascade="all, delete-orphan")
    ingestions = relationship("IngestionRecord", back_populates="document",
                              cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    char_count = Column(Integer, default=0)
    vector_id = Column(String, index=True)  # ChromaDB ID reference
    metadata_json = Column(JSON, default=dict)

    document = relationship("Document", back_populates="chunks")

class IngestionRecord(Base):
    __tablename__ = "ingestion_records"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    strategy = Column(String(50), nullable=False)  # recursive, semantic, etc.
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="pending")  # pending, running, success, failed
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    document = relationship("Document", back_populates="ingestions")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=_uuid)
    title = Column(String(200), default="New Chat")
    model_used = Column(String(100), default="llama3.1:8b")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    messages = relationship("ChatMessage", back_populates="session",
                            cascade="all, delete-orphan",
                            order_by="ChatMessage.created_at")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    role = Column(String(10), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    sources_json = Column(JSON, nullable=True)  # Retrieved chunks used
    tokens_used = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)

    session = relationship("ChatSession", back_populates="messages")
```

---

## 4. PDS Repository (Async CRUD)

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, delete, func
from pathlib import Path
import hashlib
import os

class PDSRepository:
    """Async repository for Personal Data Store operations."""

    def __init__(self, db_path: str = "./data/pds.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_session(self) -> AsyncSession:
        """Get a new async session."""
        return self.session_factory()

    # ── Document CRUD ──

    async def add_document(
        self,
        file_path: str,
        filename: str,
        file_type: str,
        category: str = "general",
        tags: str = "",
        description: str = ""
    ) -> Document:
        """Register a document in the PDS."""
        file_size = os.path.getsize(file_path)
        file_hash = hashlib.md5(open(file_path, "rb").read()).hexdigest()

        # Check for duplicate (same hash)
        async with self.get_session() as session:
            result = await session.execute(
                select(Document).where(Document.file_hash == file_hash)
            )
            if result.scalar_one_or_none():
                raise ValueError(f"Duplicate document (hash: {file_hash})")

            doc = Document(
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                file_size_bytes=file_size,
                file_hash=file_hash,
                category=category,
                tags=tags,
                description=description,
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            return doc

    async def get_document(self, doc_id: str) -> Document | None:
        async with self.get_session() as session:
            return await session.get(Document, doc_id)

    async def get_all_documents(
        self,
        category: str | None = None,
        file_type: str | None = None,
        limit: int = 50,
        offset: int = 0
    ) -> list[Document]:
        async with self.get_session() as session:
            query = select(Document)
            if category:
                query = query.where(Document.category == category)
            if file_type:
                query = query.where(Document.file_type == file_type)
            query = query.offset(offset).limit(limit).order_by(Document.uploaded_at.desc())
            result = await session.execute(query)
            return result.scalars().all()

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document and its file from disk."""
        async with self.get_session() as session:
            doc = await session.get(Document, doc_id)
            if not doc:
                return False

            # Delete file from disk
            if os.path.exists(doc.file_path):
                os.remove(doc.file_path)

            # Delete from ChromaDB (caller's responsibility)
            # Delete from PDS (cascades to chunks and ingestions)
            await session.delete(doc)
            await session.commit()
            return True

    async def get_document_stats(self) -> dict:
        async with self.get_session() as session:
            total = await session.scalar(select(func.count(Document.id)))
            total_chars = await session.scalar(
                select(func.coalesce(func.sum(Document.char_count), 0))
            )
            by_type = await session.execute(
                select(Document.file_type, func.count(Document.id))
                .group_by(Document.file_type)
            )
            return {
                "total_documents": total,
                "total_chars": total_chars,
                "documents_by_type": dict(by_type.all()),
            }

    # ── Ingestion Tracking ──

    async def create_ingestion(
        self, document_id: str, strategy: str
    ) -> IngestionRecord:
        async with self.get_session() as session:
            record = IngestionRecord(
                document_id=document_id,
                strategy=strategy,
                status="pending",
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def update_ingestion_status(
        self, ingestion_id: str, status: str,
        chunk_count: int = 0, error_message: str = ""
    ):
        async with self.get_session() as session:
            record = await session.get(IngestionRecord, ingestion_id)
            if record:
                record.status = status
                record.chunk_count = chunk_count
                if error_message:
                    record.error_message = error_message
                if status in ("success", "failed"):
                    record.completed_at = _utcnow()
                await session.commit()

    # ── Chat History ──

    async def create_session(self, title: str = "New Chat",
                             model: str = "llama3.1:8b") -> ChatSession:
        async with self.get_session() as session:
            chat = ChatSession(title=title, model_used=model)
            session.add(chat)
            await session.commit()
            await session.refresh(chat)
            return chat

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict] = None,
        tokens: int = 0,
        latency: int = 0
    ) -> ChatMessage:
        async with self.get_session() as session:
            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                sources_json=sources,
                tokens_used=tokens,
                latency_ms=latency,
            )
            session.add(msg)
            await session.commit()
            await session.refresh(msg)
            return msg

    async def get_session_messages(
        self, session_id: str, limit: int = 100
    ) -> list[ChatMessage]:
        async with self.get_session() as session:
            result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
                .limit(limit)
            )
            return result.scalars().all()

    # ── Cleanup ──

    async def close(self):
        await self.engine.dispose()
```

---

## 5. File Store Manager

Manages physical file storage on disk.

```python
import shutil
from pathlib import Path

class FileStore:
    """Manages file storage on disk."""

    def __init__(self, base_path: str = "./data/documents"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def store_upload(self, file_bytes: bytes, filename: str) -> str:
        """
        Store an uploaded file and return its path.
        
        Organizes by date: data/documents/YYYY-MM/unique_filename
        """
        from datetime import date
        date_dir = self.base_path / str(date.today())
        date_dir.mkdir(exist_ok=True)

        # Ensure unique filename
        import uuid
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        file_path = date_dir / unique_name

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        return str(file_path)

    def read_file(self, file_path: str) -> str | None:
        """Read text file content."""
        path = Path(file_path)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None  # Binary file, needs special handling

    def delete_file(self, file_path: str) -> bool:
        """Delete a stored file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except OSError:
            pass
        return False

    def get_storage_usage(self) -> dict:
        """Get storage statistics."""
        total_size = 0
        file_count = 0
        for f in self.base_path.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
                file_count += 1
        return {
            "total_files": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "base_path": str(self.base_path),
        }
```

---

## 6. Supported File Types

| Type | Extension | Text Extraction | Chunking Strategy |
|------|-----------|----------------|-------------------|
| Plain Text | `.txt` | Direct read | Recursive |
| Markdown | `.md` | Direct read | Markdown (heading-aware) |
| Python | `.py` | Direct read | Token-based |
| JavaScript | `.js` | Direct read | Token-based |
| CSV | `.csv` | Direct read | Sentence |
| JSON | `.json` | Direct read | Recursive |
| PDF | `.pdf` | PyMuPDF / pdfplumber | Recursive |
| DOCX | `.docx` | python-docx | Recursive |
| HTML | `.html` | BeautifulSoup | Markdown-like |

### PDF Text Extraction (Free)

```python
class PDFExtractor:
    """Extract text from PDFs using free libraries."""

    @staticmethod
    def extract(file_path: str) -> tuple[str, list[dict]]:
        """
        Extract text and page metadata from a PDF.
        
        Returns:
            (full_text, page_metadata)
        """
        import fitz  # PyMuPDF — free and fast
        
        doc = fitz.open(file_path)
        full_text = []
        pages = []
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            full_text.append(text)
            pages.append({
                "page": page_num,
                "char_count": len(text),
            })
        
        doc.close()
        return "\n\n".join(full_text), pages
```

---

## 7. Sync and Backup

### Document Sync (Re-indexing)

```python
class DocumentSync:
    """
    Detect and sync changes between file system and PDS.
    
    When a file is modified, this detects the change via file hash
    and triggers re-ingestion.
    """

    def __init__(self, pds: PDSRepository, vector_store):
        self.pds = pds
        self.vector_store = vector_store

    async def check_for_updates(self) -> list[dict]:
        """
        Scan all registered documents and detect changes.
        
        Returns:
            List of changed documents with action needed.
        """
        changes = []
        docs = await self.pds.get_all_documents()

        for doc in docs:
            if not os.path.exists(doc.file_path):
                changes.append({
                    "doc_id": doc.id,
                    "filename": doc.filename,
                    "action": "missing",
                })
                continue

            # Re-hash file
            with open(doc.file_path, "rb") as f:
                current_hash = hashlib.md5(f.read()).hexdigest()

            if current_hash != doc.file_hash:
                changes.append({
                    "doc_id": doc.id,
                    "filename": doc.filename,
                    "action": "reindex",
                    "old_hash": doc.file_hash,
                    "new_hash": current_hash,
                })

        return changes

    async def reindex_document(self, doc_id: str):
        """Re-index a document that has changed."""
        doc = await self.pds.get_document(doc_id)
        if not doc:
            return

        # Delete old vectors
        self.vector_store.delete_document(doc.filename)

        # Re-read and re-chunk
        content = FileStore().read_file(doc.file_path)
        if not content:
            return

        # Update file hash
        with open(doc.file_path, "rb") as f:
            doc.file_hash = hashlib.md5(f.read()).hexdigest()

        # Re-ingest (pipeline handles chunking + embedding + storage)
        return content
```

### Backup Strategy

```python
async def backup_pds(pds_path: str, backup_dir: str):
    """Backup the entire PDS (SQLite + ChromaDB + documents)."""
    import shutil
    from datetime import date
    
    timestamp = str(date.today())
    backup_path = Path(backup_dir) / f"rag-backup-{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Backup SQLite database
    db_file = Path(pds_path)
    if db_file.exists():
        shutil.copy2(db_file, backup_path / "pds.db")
    
    # 2. Backup ChromaDB
    chroma_dir = Path("./data/chroma_db")
    if chroma_dir.exists():
        shutil.copytree(chroma_dir, backup_path / "chroma_db")
    
    # 3. Backup documents
    docs_dir = Path("./data/documents")
    if docs_dir.exists():
        shutil.copytree(docs_dir, backup_path / "documents")
    
    print(f"Backup saved to {backup_path}")
    return str(backup_path)


async def restore_pds(backup_path: str):
    """Restore PDS from a backup."""
    backup = Path(backup_path)
    
    if (backup / "pds.db").exists():
        shutil.copy2(backup / "pds.db", "./data/pds.db")
    if (backup / "chroma_db").exists():
        if Path("./data/chroma_db").exists():
            shutil.rmtree("./data/chroma_db")
        shutil.copytree(backup / "chroma_db", "./data/chroma_db")
    if (backup / "documents").exists():
        if Path("./data/documents").exists():
            shutil.rmtree("./data/documents")
        shutil.copytree(backup / "documents", "./data/documents")
    
    print(f"Restored from {backup_path}")
```

---

## 8. Configuration Store

```python
from pydantic_settings import BaseSettings

class PDSConfig(BaseSettings):
    """PDS configuration — type-safe settings with env var override."""
    
    pds_db_path: str = "./data/pds.db"
    documents_path: str = "./data/documents"
    chroma_persist_path: str = "./data/chroma_db"
    
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64
    max_file_size_mb: int = 50
    
    allowed_file_types: list[str] = [
        ".txt", ".md", ".py", ".js", ".csv", ".json",
        ".pdf", ".docx", ".html", ".xml", ".yaml", ".yml"
    ]
    
    class Config:
        env_prefix = "PDS_"
        env_file = ".env"
```

---

## 9. PDS Checklist

- [ ] SQLite database created (`pds.db`)
- [ ] All tables created (documents, chunks, ingestions, chat)
- [ ] `PDSRepository` initialized and tested
- [ ] Document add/get/delete operations work
- [ ] File upload → PDS registration → vector store → complete flow tested
- [ ] Duplicate detection by file hash working
- [ ] Chat session create/add-message/get-history tested
- [ ] File store manager organizes files by date
- [ ] PDF text extraction working (PyMuPDF installed)
- [ ] Backup/restore commands tested
- [ ] Error handling for missing files, duplicate files, large files
- [ ] Storage usage reporting works
- [ ] Configuration loaded from environment variables
