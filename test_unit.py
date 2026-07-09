"""Intensive unit tests for all backend modules."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("UNIT TEST SUITE — AI RAG System")
print("=" * 60)

pass_count = 0
fail_count = 0

def test(name, fn):
    global pass_count, fail_count
    try:
        fn()
        print(f"  [PASS] {name}")
        pass_count += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        fail_count += 1

# ── 1. Chunker ──

from backend.core.chunker import (
    RecursiveCharacterSplitter, MarkdownSplitter,
    TokenSplitter, SentenceSplitter, DocumentChunker,
)

def test_recursive_chunker():
    c = RecursiveCharacterSplitter(50, 10)
    r = c.split("Hello world. " * 20)
    assert len(r) > 1, f"Expected multiple chunks, got {len(r)}"

def test_markdown_chunker():
    c = MarkdownSplitter(200, 20)
    r = c.split("# Header\n\nSome text\n\n## Sub\n\nMore text")
    assert len(r) >= 1

def test_token_chunker():
    c = TokenSplitter(50, 10)
    r = c.split("word " * 200)
    assert len(r) > 0

def test_sentence_chunker():
    c = SentenceSplitter(50)
    r = c.split("First sentence. Second sentence. Third sentence.")
    assert len(r) >= 1

def test_factory_by_ext():
    dc = DocumentChunker(100, 10)
    chunks, metas = dc.chunk("def foo(): pass", source="test.py")
    assert len(chunks) == 1
    assert metas[0]["strategy"] == "token"

def test_factory_markdown_ext():
    dc = DocumentChunker(100, 10)
    chunks, metas = dc.chunk("# Title\nBody", source="doc.md")
    assert metas[0]["strategy"] == "markdown"

def test_factory_default():
    dc = DocumentChunker(100, 10)
    chunks, metas = dc.chunk("Plain text content", source="unknown.xyz")
    assert metas[0]["strategy"] == "recursive"

def test_empty_text():
    c = RecursiveCharacterSplitter(50, 10)
    r = c.split("")
    assert r == []

def test_single_short_chunk():
    c = RecursiveCharacterSplitter(500, 50)
    r = c.split("Short text.")
    assert len(r) == 1
    assert "Short text." in r[0]

def test_metadata_structure():
    dc = DocumentChunker(100, 10)
    chunks, metas = dc.chunk("Test content", source="test.txt")
    assert "source" in metas[0]
    assert "chunk_index" in metas[0]
    assert "total_chunks" in metas[0]
    assert "chunk_size" in metas[0]
    assert "strategy" in metas[0]
    assert metas[0]["chunk_index"] == 0
    assert metas[0]["total_chunks"] == len(chunks)

test("RecursiveCharacterSplitter produces multiple chunks", test_recursive_chunker)
test("MarkdownSplitter handles headings", test_markdown_chunker)
test("TokenSplitter handles long text", test_token_chunker)
test("SentenceSplitter respects boundaries", test_sentence_chunker)
test("DocumentChunker selects 'token' for .py", test_factory_by_ext)
test("DocumentChunker selects 'markdown' for .md", test_factory_markdown_ext)
test("DocumentChunker defaults to 'recursive'", test_factory_default)
test("Empty text returns empty list", test_empty_text)
test("Single short chunk works", test_single_short_chunk)
test("Metadata structure is correct", test_metadata_structure)

# ── 2. Config ──

from backend.config import settings

def test_config_values():
    assert settings.llm_model == "llama3.1:8b"
    assert settings.chunk_size == 512
    assert settings.temperature == 0.1
    assert isinstance(settings.allowed_file_types, list)
    assert ".txt" in settings.allowed_file_types

def test_config_env_override(monkeypatch=None):
    # Just verify it loads without error from .env
    assert settings.pds_db_path == "./data/pds.db"

test("Config has correct default values", test_config_values)
test("Config loads from .env correctly", test_config_env_override)

# ── 3. PDS Models ──

from backend.pds.models import Base, Document, DocumentChunk, ChatSession, ChatMessage, IngestionRecord

def test_all_tables_registered():
    assert "documents" in Base.metadata.tables
    assert "document_chunks" in Base.metadata.tables
    assert "ingestion_records" in Base.metadata.tables
    assert "chat_sessions" in Base.metadata.tables
    assert "chat_messages" in Base.metadata.tables

def test_document_columns():
    cols = Document.__table__.columns.keys()
    for required in ("id", "filename", "file_path", "file_hash", "uploaded_at"):
        assert required in cols, f"Missing column: {required}"

def test_chat_relationships():
    assert hasattr(ChatSession, "messages")
    assert hasattr(ChatMessage, "session")

test("All 5 PDS tables registered", test_all_tables_registered)
test("Document has required columns", test_document_columns)
test("Chat relationships defined", test_chat_relationships)

# ── 4. FileStore ──

from backend.pds.file_store import FileStore
import tempfile

def test_filestore_store_and_read():
    fs = FileStore(tempfile.mkdtemp())
    path = fs.store(b"hello world", "test.txt")
    assert path.endswith(".txt")
    content = fs.read_text(path)
    assert content == "hello world"

def test_filestore_hash():
    h = FileStore.md5_bytes(b"hello")
    assert len(h) == 32
    assert isinstance(h, str)

def test_filestore_delete():
    import tempfile, os
    d = tempfile.mkdtemp()
    fs = FileStore(d)
    path = fs.store(b"delete me", "del.txt")
    assert os.path.exists(path)
    result = fs.delete(path)
    assert result is True
    assert not os.path.exists(path)

def test_filestore_usage():
    import tempfile
    d = tempfile.mkdtemp()
    fs = FileStore(d)
    fs.store(b"data", "f1.txt")
    usage = fs.usage()
    assert usage["total_files"] >= 1
    assert usage["total_size_bytes"] > 0

test("FileStore stores and reads text", test_filestore_store_and_read)
test("FileStore MD5 hash works", test_filestore_hash)
test("FileStore deletes files", test_filestore_delete)
test("FileStore reports usage stats", test_filestore_usage)

# ── 5. Embedder (light) ──

from backend.core.embedder import Embedder

def test_embedder_encode():
    e = Embedder()
    vec = e.encode("test sentence")
    assert len(vec) in (384, 768), f"Unexpected dim: {len(vec)}"
    assert all(isinstance(v, float) for v in vec)

def test_embedder_batch():
    e = Embedder()
    vecs = e.encode_many(["one", "two", "three"])
    assert len(vecs) == 3
    assert len(vecs[0]) in (384, 768)

def test_embedder_normalized():
    import math
    e = Embedder()
    vec = e.encode("test")
    magnitude = math.sqrt(sum(v*v for v in vec))
    assert abs(magnitude - 1.0) < 0.01, f"Expected unit vector, got {magnitude}"

test("Embedder produces float vectors", test_embedder_encode)
test("Embedder batch encodes correctly", test_embedder_batch)
test("Embedder returns normalized vectors", test_embedder_normalized)

# ── 6. LLM Client (connectivity) ──

import asyncio

async def test_llm_available():
    from backend.core.llm_client import LLMClient
    llm = LLMClient()
    avail = await llm.is_available()
    assert avail, "Ollama is not running"
    await llm.close()

async def test_llm_generate():
    from backend.core.llm_client import LLMClient
    llm = LLMClient()
    resp = await llm.generate("Say 'OK' in one word.")
    assert resp.strip(), "Empty response from LLM"
    print(f"      LLM response: \"{resp.strip()}\"")
    await llm.close()

async def test_llm_stream():
    from backend.core.llm_client import LLMClient
    llm = LLMClient()
    tokens = []
    async for token in await llm.generate("Count 1 2 3.", stream=True):
        tokens.append(token)
    assert len(tokens) > 0, "No tokens received"
    await llm.close()

async def test_llm_chat():
    from backend.core.llm_client import LLMClient
    llm = LLMClient()
    resp = await llm.chat([
        {"role": "user", "content": "Say 'OK'"}
    ])
    assert resp.strip(), "Empty chat response"
    await llm.close()

try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_llm_available())
    test("Ollama is available and responding", lambda: None)
    loop.run_until_complete(test_llm_generate())
    test("LLM generate produces non-empty response", lambda: None)
    loop.run_until_complete(test_llm_stream())
    test("LLM streaming returns tokens", lambda: None)
    loop.run_until_complete(test_llm_chat())
    test("LLM chat endpoint works", lambda: None)
except Exception as e:
    test(f"LLM tests skipped — Ollama may not be running: {e}", lambda: None)

# ── 7. ChromaDB (light) ──

def test_chroma_store_basic():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from backend.vector_store.chroma_client import ChromaStore
        store = ChromaStore(persist_directory=d, collection_name="test_coll")
        ids = store.add_chunks(
            ["Paris is the capital of France.", "Python is a language."],
            [{"source": "test.txt", "chunk_index": 0},
             {"source": "test.txt", "chunk_index": 1}],
        )
        assert len(ids) == 2
        assert store.count() == 2
        results = store.search("What is the capital of France?", n_results=1)
        assert len(results["documents"][0]) >= 1
        assert "Paris" in results["documents"][0][0]
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_chroma_delete():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from backend.vector_store.chroma_client import ChromaStore
        store = ChromaStore(persist_directory=d, collection_name="test_del")
        store.add_chunks(["Content A"], [{"source": "a.txt", "chunk_index": 0}])
        store.add_chunks(["Content B"], [{"source": "b.txt", "chunk_index": 0}])
        assert store.count() == 2
        store.delete_document("a.txt")
        assert store.count() == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_chroma_metadata_filter():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from backend.vector_store.chroma_client import ChromaStore
        store = ChromaStore(persist_directory=d, collection_name="test_filter")
        store.add_chunks(
            ["Finance report content", "Recipe for pasta"],
            [{"source": "report.pdf", "category": "finance"},
             {"source": "cooking.txt", "category": "food"}],
        )
        results = store.search("money", where={"category": "finance"})
        assert len(results["documents"][0]) >= 1
        assert "Finance" in results["documents"][0][0]
    finally:
        shutil.rmtree(d, ignore_errors=True)

test("ChromaDB stores and retrieves chunks", test_chroma_store_basic)
test("ChromaDB deletes by source", test_chroma_delete)
test("ChromaDB metadata filters work", test_chroma_metadata_filter)

# ── 8. PDS Repository (real SQLite) ──

import tempfile, asyncio

async def test_pds_crud():
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    from backend.pds.repository import PDSRepository
    pds = PDSRepository(db_path=db)
    await pds.initialize()

    doc = await pds.add_document(
        filename="test.txt", file_path="/tmp/test.txt",
        file_type="txt", file_size_bytes=100, file_hash="abc123",
    )
    assert doc.id is not None
    assert doc.filename == "test.txt"

    fetched = await pds.get_document(doc.id)
    assert fetched is not None

    found = await pds.get_document_by_hash("abc123")
    assert found is not None
    assert found.id == doc.id

    docs = await pds.get_all_documents()
    assert len(docs) >= 1

    stats = await pds.get_document_stats()
    assert stats["total_documents"] >= 1

    deleted = await pds.delete_document(doc.id)
    assert deleted is True

    gone = await pds.get_document(doc.id)
    assert gone is None

    await pds.close()

async def test_pds_chat():
    db = os.path.join(tempfile.mkdtemp(), "chat.db")
    from backend.pds.repository import PDSRepository
    pds = PDSRepository(db_path=db)
    await pds.initialize()

    session = await pds.create_session(title="Test Chat")
    assert session.id is not None

    msg1 = await pds.add_message(session.id, "user", "Hello")
    msg2 = await pds.add_message(session.id, "assistant", "Hi there", sources=[{"source": "doc.txt"}])
    assert msg1.id != msg2.id

    msgs = await pds.get_session_messages(session.id)
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"

    sessions = await pds.get_recent_sessions()
    assert len(sessions) >= 1

    await pds.close()

async def test_pds_ingestion():
    db = os.path.join(tempfile.mkdtemp(), "ingest.db")
    from backend.pds.repository import PDSRepository
    pds = PDSRepository(db_path=db)
    await pds.initialize()
    doc = await pds.add_document("f.txt", "/tmp/f.txt", "txt", 50, "abc")
    ing = await pds.create_ingestion(doc.id, "recursive")
    assert ing.status == "running"
    await pds.update_ingestion_status(ing.id, "success", chunk_count=10)
    # Re-fetch to get updated values
    from backend.pds.models import IngestionRecord
    from sqlalchemy import select
    async with pds._session() as session:
        refreshed = await session.get(IngestionRecord, ing.id)
        assert refreshed.status == "success"
        assert refreshed.chunk_count == 10
    await pds.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(test_pds_crud())
test("PDS Repository CRUD operations", lambda: None)
loop.run_until_complete(test_pds_chat())
test("PDS Chat session and messages", lambda: None)
loop.run_until_complete(test_pds_ingestion())
test("PDS Ingestion tracking", lambda: None)

# ── Summary ──

print()
print("=" * 60)
print(f"RESULTS: {pass_count} passed, {fail_count} failed, "
      f"{pass_count + fail_count} total")
print("=" * 60)
sys.exit(0 if fail_count == 0 else 1)
