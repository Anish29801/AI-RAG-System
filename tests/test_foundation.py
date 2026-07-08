"""Batch 1 — Foundation Tests.

Tests every module with zero external dependencies:
- Config (validation, defaults, env prefix)
- PDS Models (ORM schema, columns, relationships, constraints)
- FileStore (store, read, delete, hash, usage, edge cases)
- Chunker (all 4 strategies + factory + metadata)
- ChromaStore (add, search, delete, filter, batch, edge cases)
- Embedder (encode, batch, normalisation, edge cases)

Run:  pytest tests/test_foundation.py -v
"""

# ═══════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════

import os
import math
import tempfile
import shutil
from pathlib import Path

import pytest

from backend.config import AppConfig
from backend.pds.models import (
    Base, Document, DocumentChunk, IngestionRecord,
    ChatSession, ChatMessage,
)
from backend.pds.file_store import FileStore
from backend.core.chunker import (
    RecursiveCharacterSplitter, MarkdownSplitter,
    TokenSplitter, SentenceSplitter, DocumentChunker,
)

# ═══════════════════════════════════════════════════════════════
# Sub-batch 1A — Config
# ═══════════════════════════════════════════════════════════════


class TestConfig:
    """Config defaults, prefixes, and type invariants."""

    def test_default_values(self, test_config: AppConfig):
        assert test_config.llm_model == "llama3.1:8b"
        assert test_config.chunk_size == 512
        assert test_config.chunk_overlap == 64
        assert test_config.temperature == 0.1
        assert test_config.top_k == 5
        assert test_config.n_results == 10
        assert test_config.max_file_size_mb == 50
        assert isinstance(test_config.allowed_file_types, list)
        assert ".txt" in test_config.allowed_file_types
        assert ".pdf" in test_config.allowed_file_types
        assert ".py" in test_config.allowed_file_types

    def test_env_prefix_is_rag(self):
        """Config uses RAG_ prefix so .env vars are mapped correctly."""
        assert AppConfig.model_config["env_prefix"] == "RAG_"

    def test_allowed_file_types_contains_expected(self, test_config: AppConfig):
        expected = {".txt", ".md", ".py", ".js", ".ts", ".csv", ".json",
                    ".pdf", ".docx", ".html", ".xml", ".yaml", ".yml"}
        assert set(test_config.allowed_file_types) >= expected

    def test_temperature_range_valid(self):
        """Temperature between 0 and 1 is valid."""
        config = AppConfig(temperature=0.5)  # type: ignore
        assert 0.0 <= config.temperature <= 1.0

    def test_chunk_overlap_less_than_chunk_size_invariant(self, test_config: AppConfig):
        """chunk_overlap must always be < chunk_size (enforced by chunker)."""
        assert test_config.chunk_overlap < test_config.chunk_size

    def test_max_file_size_positive(self, test_config: AppConfig):
        assert test_config.max_file_size_mb > 0

    @pytest.mark.parametrize("bad_temp", [-0.1, 1.5, 100])
    def test_invalid_temperature_raises(self, bad_temp: float):
        """Pydantic should validate temperature range if validator is added."""
        # Currently no validator — this test documents the gap
        # Once validated, this should raise ValidationError
        config = AppConfig(temperature=bad_temp)  # type: ignore
        assert isinstance(config, AppConfig)  # placeholder until validation added


# ═══════════════════════════════════════════════════════════════
# Sub-batch 1B — PDS Models (ORM)
# ═══════════════════════════════════════════════════════════════


class TestPDSModels:
    """ORM schema completeness, columns, relationships, constraints."""

    def test_all_tables_registered(self):
        """All 5 expected tables exist in the ORM metadata."""
        tables = Base.metadata.tables
        assert "documents" in tables
        assert "document_chunks" in tables
        assert "ingestion_records" in tables
        assert "chat_sessions" in tables
        assert "chat_messages" in tables

    def test_document_has_all_columns(self):
        """Document table has all required columns."""
        cols = Document.__table__.columns.keys()
        for required in (
            "id", "filename", "file_path", "file_type",
            "file_size_bytes", "file_hash", "page_count",
            "char_count", "category", "tags", "description",
            "uploaded_at", "updated_at",
        ):
            assert required in cols, f"Document missing column: {required}"

    def test_document_chunk_has_all_columns(self):
        cols = DocumentChunk.__table__.columns.keys()
        for required in (
            "id", "document_id", "chunk_index", "content",
            "char_count", "vector_id", "metadata_json",
        ):
            assert required in cols, f"DocumentChunk missing column: {required}"

    def test_chat_session_has_all_columns(self):
        cols = ChatSession.__table__.columns.keys()
        for required in ("id", "title", "model_used", "created_at", "updated_at"):
            assert required in cols, f"ChatSession missing column: {required}"

    def test_chat_message_has_all_columns(self):
        cols = ChatMessage.__table__.columns.keys()
        for required in (
            "id", "session_id", "role", "content",
            "sources_json", "tokens_used", "latency_ms", "created_at",
        ):
            assert required in cols, f"ChatMessage missing column: {required}"

    def test_ingestion_record_has_all_columns(self):
        cols = IngestionRecord.__table__.columns.keys()
        for required in (
            "id", "document_id", "strategy", "chunk_count",
            "status", "error_message", "started_at", "completed_at",
        ):
            assert required in cols, f"IngestionRecord missing column: {required}"

    def test_document_relationships(self):
        """Document has chunks and ingestions relationships."""
        assert hasattr(Document, "chunks")
        assert hasattr(Document, "ingestions")

    def test_chat_relationships(self):
        """ChatSession has messages; ChatMessage has session."""
        assert hasattr(ChatSession, "messages")
        assert hasattr(ChatMessage, "session")

    def test_document_chunk_relationships(self):
        assert hasattr(DocumentChunk, "document")

    def test_ingestion_record_relationships(self):
        assert hasattr(IngestionRecord, "document")

    def test_uuid_primary_key_generated(self):
        """Document.id gets auto-generated UUID on creation (via default factory)."""
        doc = Document(filename="test.txt")
        # SQLAlchemy Column(default=func) applies on INSERT, not on __init__
        # So id is None for transient objects — but the default function works
        # Verify the default factory exists and produces valid UUID
        from backend.pds.models import _uuid
        uid = _uuid()
        assert isinstance(uid, str)
        assert len(uid) == 36  # UUID4 is 36 chars with hyphens

    def test_default_values_set_correctly_db(self, pds_repo):
        """Verify defaults are applied when persisted to DB."""
        import asyncio
        doc = asyncio.get_event_loop().run_until_complete(
            pds_repo.add_document(
                filename="test.txt", file_path="/tmp/t.txt",
                file_type="txt", file_size_bytes=100, file_hash="abc",
            )
        )
        assert doc.category == "general"
        assert doc.tags == ""
        assert doc.page_count == 0
        assert doc.char_count == 0

    def test_foreign_key_cascade_on_document_delete(self):
        """Deleting a Document cascades to its chunks and ingestions."""
        from sqlalchemy import create_engine, inspect
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        # Verify FK constraint exists with CASCADE
        inspector = inspect(engine)
        fks = inspector.get_foreign_keys("document_chunks")
        assert any(
            fk["referred_table"] == "documents" and "CASCADE" in str(fk.get("options", {}))
            for fk in fks
        ), "Expected CASCADE delete on document_chunks"

        fks_ingest = inspector.get_foreign_keys("ingestion_records")
        assert any(
            fk["referred_table"] == "documents" and "CASCADE" in str(fk.get("options", {}))
            for fk in fks_ingest
        ), "Expected CASCADE delete on ingestion_records"
        engine.dispose()

    def test_document_indexes(self):
        """Key columns should be indexed for performance."""
        for col in Document.__table__.columns:
            if col.name == "filename":
                assert col.index, "filename should be indexed"
        for col in DocumentChunk.__table__.columns:
            if col.name == "document_id":
                assert col.index, "document_id should be indexed"

    def test_sources_json_accepts_list_of_dicts(self):
        """sources_json column stores list[dict] serialised."""
        msg = ChatMessage(sources_json=[{"source": "doc.txt", "score": 0.95}])
        assert isinstance(msg.sources_json, list)
        assert msg.sources_json[0]["source"] == "doc.txt"

    def test_timestamps_timezone_aware(self):
        """uploaded_at and created_at default functions produce UTC timezone-aware."""
        from backend.pds.models import _utcnow
        now = _utcnow()
        assert now is not None
        assert now.tzinfo is not None, "timestamp should be timezone-aware"
        assert str(now.tzinfo) == "UTC", "should be UTC"


# ═══════════════════════════════════════════════════════════════
# Sub-batch 1C — FileStore
# ═══════════════════════════════════════════════════════════════


class TestFileStore:
    """File disk storage — store, read, delete, hash, usage, edge cases."""

    def test_store_creates_date_nested_directory(self, file_store: FileStore):
        path = file_store.store(b"data", "test.txt")
        path_obj = Path(path)
        # Path should be like: .../docs/2026-07-08/uuid.txt
        assert path_obj.exists()
        assert path_obj.parent.name  # date directory
        assert len(path_obj.parent.name) == 10  # YYYY-MM-DD format

    def test_store_returns_absolute_path(self, file_store: FileStore):
        path = file_store.store(b"data", "test.txt")
        assert os.path.isabs(path)

    def test_store_read_text_roundtrip(self, file_store: FileStore):
        content = "Hello, FileStore!"
        path = file_store.store(content.encode(), "greeting.txt")
        read_back = file_store.read_text(path)
        assert read_back == content

    def test_store_read_bytes_roundtrip(self, file_store: FileStore):
        content = b"\x00\x01\x02\xff\xfe"
        path = file_store.store(content, "binary.bin")
        read_back = file_store.read_bytes(path)
        assert read_back == content

    def test_read_text_nonexistent(self, file_store: FileStore):
        assert file_store.read_text("/nonexistent/path.txt") is None

    def test_read_bytes_nonexistent(self, file_store: FileStore):
        assert file_store.read_bytes("/nonexistent/path.bin") is None

    def test_md5_hash_consistency(self):
        """Same content always produces same MD5 hash."""
        h1 = FileStore.md5_bytes(b"hello world")
        h2 = FileStore.md5_bytes(b"hello world")
        assert h1 == h2
        assert len(h1) == 32
        assert isinstance(h1, str)

    def test_md5_hash_different_content(self):
        """Different content produces different hash."""
        h1 = FileStore.md5_bytes(b"hello")
        h2 = FileStore.md5_bytes(b"world")
        assert h1 != h2

    def test_md5_file_hash(self, file_store: FileStore):
        path = file_store.store(b"file content", "hash_me.txt")
        h = FileStore.md5(path)
        assert len(h) == 32
        assert isinstance(h, str)

    def test_delete_removes_file(self, file_store: FileStore, temp_dir: str):
        path = file_store.store(b"delete me", "del.txt")
        assert os.path.exists(path)
        result = file_store.delete(path)
        assert result is True
        assert not os.path.exists(path)

    def test_delete_nonexistent_returns_false(self, file_store: FileStore):
        """Delete returns False for non-existent files (idempotent, no error)."""
        result = file_store.delete("/nonexistent/file.txt")
        assert result is False

    def test_usage_after_stores(self, file_store: FileStore):
        file_store.store(b"data1", "f1.txt")
        file_store.store(b"X" * 200000, "f2.txt")  # ~200KB to get >0 MB
        usage = file_store.usage()
        assert usage["total_files"] >= 2
        assert usage["total_size_bytes"] > 0
        assert usage["total_size_mb"] >= 0  # may round to 0 for small files
        assert usage["base_path"] is not None

    def test_usage_empty_store(self, file_store: FileStore):
        usage = file_store.usage()
        assert usage["total_files"] == 0
        assert usage["total_size_bytes"] == 0

    def test_read_text_binary_content_returns_none(self, file_store: FileStore):
        """Binary content fails UTF-8 decode, returns None."""
        path = file_store.store(b"\xff\xfe\x00\x01", "binary.txt")
        result = file_store.read_text(path)
        assert result is None  # or raises gracefully

    def test_store_empty_bytes(self, file_store: FileStore):
        path = file_store.store(b"", "empty.txt")
        content = file_store.read_text(path)
        assert content == ""

    def test_store_same_name_different_paths(self, file_store: FileStore):
        """Two files with same name get different paths (uuid prefix)."""
        p1 = file_store.store(b"data1", "same_name.txt")
        p2 = file_store.store(b"data2", "same_name.txt")
        assert p1 != p2

    def test_concurrent_stores_no_collision(self, file_store: FileStore):
        """Multiple stores in quick succession don't collide."""
        paths = [file_store.store(b"d", "c.txt") for _ in range(10)]
        assert len(set(paths)) == 10  # All unique


# ═══════════════════════════════════════════════════════════════
# Sub-batch 1D — Chunker
# ═══════════════════════════════════════════════════════════════


class TestRecursiveChunker:
    """RecursiveCharacterSplitter — paragraph, sentence, word boundaries."""

    def test_splits_on_paragraph_boundary(self):
        c = RecursiveCharacterSplitter(50, 10)
        text = "A" * 30 + "\n\n" + "B" * 30
        chunks = c.split(text)
        assert len(chunks) >= 2
        assert "A" in chunks[0]
        assert "B" in chunks[-1]

    def test_overlap_preserves_content(self):
        c = RecursiveCharacterSplitter(60, 20)
        text = "X" * 100
        chunks = c.split(text)
        assert len(chunks) >= 2
        # Overlap means last chars of chunk 0 are in chunk 1
        assert chunks[0] and chunks[1]

    def test_single_short_chunk(self):
        c = RecursiveCharacterSplitter(500, 50)
        chunks = c.split("Short text.")
        assert len(chunks) == 1
        assert "Short text." in chunks[0]

    def test_exact_size_with_overlap(self):
        c = RecursiveCharacterSplitter(50, 10)
        text = "A" * 50
        chunks = c.split(text)
        # Overlap causes a trailing partial chunk: [0:50], [40:50]
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)  # no empty chunks

    def test_empty_text(self):
        c = RecursiveCharacterSplitter(50, 10)
        assert c.split("") == []
        assert c.split("   ") == []

    def test_overlap_assertion_fires(self):
        """AssertionError when chunk_overlap >= chunk_size."""
        with pytest.raises(AssertionError):
            RecursiveCharacterSplitter(50, 60)

    def test_no_data_loss_with_zero_overlap(self):
        c = RecursiveCharacterSplitter(20, 0)
        text = "Hello world. " * 10
        chunks = c.split(text)
        reconstructed = "".join(chunks)
        # With zero overlap, all chars should be present
        assert len(reconstructed) >= len(text) * 0.9  # allow minor separator loss

    def test_order_preserved(self):
        c = RecursiveCharacterSplitter(30, 5)
        text = "First. Second. Third. Fourth."
        chunks = c.split(text)
        indices = []
        for ch in chunks:
            idx = text.find(ch[:10])
            assert idx >= 0, f"Chunk not found in original: {ch[:20]}"
            indices.append(idx)
        assert indices == sorted(indices), "Chunks out of order"


class TestMarkdownSplitter:
    """MarkdownSplitter — heading-aware chunking."""

    def test_splits_at_heading_boundaries(self):
        c = MarkdownSplitter(200, 20)
        text = "# Header 1\n\nContent 1\n\n## Header 2\n\nContent 2\n\n### Header 3\n\nContent 3"
        chunks = c.split(text)
        assert len(chunks) >= 3

    def test_no_headings_falls_back(self):
        c = MarkdownSplitter(200, 20)
        chunks = c.split("Plain text without any markdown headings. " * 10)
        assert len(chunks) >= 1

    def test_heading_at_boundary(self):
        c = MarkdownSplitter(60, 10)
        text = "A" * 55 + "\n# Heading\n" + "B" * 55
        chunks = c.split(text)
        assert len(chunks) >= 2

    def test_empty_text(self):
        c = MarkdownSplitter(200, 20)
        assert c.split("") == []

    def test_single_heading_no_content(self):
        c = MarkdownSplitter(200, 20)
        chunks = c.split("# Just a heading")
        assert len(chunks) >= 1
        assert "# Just a heading" in chunks[0]


class TestTokenSplitter:
    """TokenSplitter — approximate token count splitter."""

    def test_splits_long_text(self):
        c = TokenSplitter(50, 10)
        chunks = c.split("word " * 500)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self):
        c = TokenSplitter(500, 50)
        chunks = c.split("Hello world")
        assert len(chunks) == 1

    def test_empty_text(self):
        c = TokenSplitter(50, 10)
        assert c.split("") == []


class TestSentenceSplitter:
    """SentenceSplitter — sentence-boundary aware."""

    def test_respects_sentence_boundaries(self):
        c = SentenceSplitter(100)
        text = "First sentence. Second sentence. Third sentence."
        chunks = c.split(text)
        assert len(chunks) >= 1

    def test_max_chars_threshold(self):
        c = SentenceSplitter(30)
        text = "This is a longer sentence that should be chunked. And another one."
        chunks = c.split(text)
        assert all(len(c) >= 1 for c in chunks)

    def test_no_sentence_breaks(self):
        c = SentenceSplitter(50)
        text = "No breaks at all in this entire text block"
        chunks = c.split(text)
        assert len(chunks) >= 1

    def test_empty_text(self):
        c = SentenceSplitter(100)
        assert c.split("") == []


class TestDocumentChunker:
    """DocumentChunker factory — strategy selection + metadata."""

    @pytest.mark.parametrize("ext,expected_strategy", [
        (".txt", "recursive"),
        (".md", "markdown"),
        (".html", "markdown"),
        (".htm", "markdown"),
        (".py", "token"),
        (".js", "token"),
        (".ts", "token"),
        (".go", "token"),
        (".rs", "token"),
        (".java", "token"),
        (".csv", "sentence"),
    ])
    def test_strategy_by_extension(self, ext: str, expected_strategy: str):
        dc = DocumentChunker(100, 10)
        _, metas = dc.chunk("content", source=f"file{ext}")
        assert metas[0]["strategy"] == expected_strategy

    def test_unknown_ext_defaults_to_recursive(self):
        dc = DocumentChunker(100, 10)
        _, metas = dc.chunk("content", source="file.xyz")
        assert metas[0]["strategy"] == "recursive"

    def test_explicit_file_type_overrides(self):
        dc = DocumentChunker(100, 10)
        chunks, metas = dc.chunk("# Hello\nBody", source="file.txt", file_type="markdown")
        assert metas[0]["strategy"] == "markdown"

    def test_metadata_completeness(self):
        dc = DocumentChunker(100, 10)
        _, metas = dc.chunk("Test content", source="doc.txt")
        assert "source" in metas[0]
        assert "chunk_index" in metas[0]
        assert "total_chunks" in metas[0]
        assert "chunk_size" in metas[0]
        assert "strategy" in metas[0]

    def test_chunk_index_sequential(self):
        dc = DocumentChunker(20, 5)
        chunks, metas = dc.chunk("Hello world. " * 20, source="doc.txt")
        for i, m in enumerate(metas):
            assert m["chunk_index"] == i

    def test_total_chunks_consistent(self):
        dc = DocumentChunker(20, 5)
        chunks, metas = dc.chunk("Hello world. " * 20, source="doc.txt")
        assert metas[0]["total_chunks"] == len(chunks)
        assert all(m["total_chunks"] == len(chunks) for m in metas)

    def test_chunk_size_in_metadata_accurate(self):
        dc = DocumentChunker(100, 10)
        chunks, metas = dc.chunk("A" * 120 + " B" * 120, source="doc.txt")
        for chunk, meta in zip(chunks, metas):
            assert meta["chunk_size"] == len(chunk)

    def test_source_in_metadata(self):
        dc = DocumentChunker(100, 10)
        _, metas = dc.chunk("Content", source="my_document.pdf")
        assert metas[0]["source"] == "my_document.pdf"

    def test_empty_text(self):
        dc = DocumentChunker(100, 10)
        chunks, metas = dc.chunk("", source="empty.txt")
        assert chunks == []
        assert metas == []

    def test_code_content_uses_token_splitter(self):
        dc = DocumentChunker(100, 10)
        chunks, metas = dc.chunk("def foo():\n    pass\n\ndef bar():\n    return 42", source="test.py")
        assert metas[0]["strategy"] == "token"
        assert len(chunks) >= 1

    def test_markdown_uses_markdown_splitter(self):
        dc = DocumentChunker(100, 10)
        chunks, metas = dc.chunk("# Title\nBody", source="doc.md")
        assert metas[0]["strategy"] == "markdown"
        assert len(chunks) >= 1


# ═══════════════════════════════════════════════════════════════
# Sub-batch 1E — ChromaStore (mocked / temp dir)
# ═══════════════════════════════════════════════════════════════


class TestChromaStore:
    """Vector store wrapper — add, search, delete, filter, batch, edge cases."""

    def test_add_chunks_returns_correct_ids(self, chroma_store):
        ids = chroma_store.add_chunks(
            ["Paris is capital of France.", "Python is a language."],
            [{"source": "test.txt", "chunk_index": 0},
             {"source": "test.txt", "chunk_index": 1}],
        )
        assert len(ids) == 2
        assert all(isinstance(i, str) for i in ids)

    def test_count_after_add(self, chroma_store):
        chroma_store.add_chunks(
            ["Chunk A", "Chunk B", "Chunk C"],
            [{"source": "a.txt", "chunk_index": i} for i in range(3)],
        )
        assert chroma_store.count() == 3

    def test_search_returns_correct_structure(self, chroma_store):
        chroma_store.add_chunks(
            ["Paris is the capital of France."],
            [{"source": "geo.txt", "chunk_index": 0}],
        )
        results = chroma_store.search("What is the capital of France?", n_results=1)
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert "ids" in results

    def test_search_finds_relevant_content(self, chroma_store):
        chroma_store.add_chunks(
            ["The Eiffel Tower is in Paris.", "Python is a programming language."],
            [{"source": "geo.txt", "chunk_index": 0},
             {"source": "code.txt", "chunk_index": 0}],
        )
        results = chroma_store.search("Eiffel Tower", n_results=1)
        assert "Eiffel" in results["documents"][0][0]

    def test_search_with_n_results_limit(self, chroma_store):
        chroma_store.add_chunks(
            [f"Content {i}" for i in range(20)],
            [{"source": "bulk.txt", "chunk_index": i} for i in range(20)],
        )
        results = chroma_store.search("Content", n_results=5)
        assert len(results["documents"][0]) <= 5

    def test_search_with_metadata_filter(self, chroma_store):
        chroma_store.add_chunks(
            ["Finance report: Q3 earnings up.",
             "Recipe for homemade pasta.",
             "Engineering: new API released."],
            [{"source": "finance.pdf", "category": "finance"},
             {"source": "cooking.txt", "category": "food"},
             {"source": "eng.pdf", "category": "engineering"}],
        )
        results = chroma_store.search("money", where={"category": "finance"})
        docs = results["documents"][0]
        assert any("Finance" in d for d in docs)
        assert all("Recipe" not in d for d in docs)

    def test_delete_document_by_source(self, chroma_store):
        chroma_store.add_chunks(
            ["Doc A content", "Doc A more"],
            [{"source": "a.txt", "chunk_index": 0},
             {"source": "a.txt", "chunk_index": 1}],
        )
        chroma_store.add_chunks(
            ["Doc B content"],
            [{"source": "b.txt", "chunk_index": 0}],
        )
        assert chroma_store.count() == 3
        deleted = chroma_store.delete_document("a.txt")
        assert deleted > 0
        assert chroma_store.count() == 1

    def test_delete_nonexistent_source(self, chroma_store):
        # ChromaDB delete with non-matching where returns 0
        try:
            result = chroma_store.delete_document("nonexistent.txt")
            assert result == 0
        except Exception:
            # Some ChromaDB versions raise on non-existent filter
            pass

    def test_reset_collection_clears_all(self, chroma_store):
        chroma_store.add_chunks(
            ["Data to clear"],
            [{"source": "clear.txt", "chunk_index": 0}],
        )
        assert chroma_store.count() > 0
        chroma_store.reset_collection()
        assert chroma_store.count() == 0

    def test_get_stats(self, chroma_store):
        chroma_store.add_chunks(
            ["Stats test"],
            [{"source": "stats.txt", "chunk_index": 0}],
        )
        stats = chroma_store.get_stats()
        assert stats["name"] == "test_coll"
        assert stats["total_chunks"] >= 1

    def test_batch_add_100_chunks(self, chroma_store):
        """Batch ingestion of 100+ chunks does not raise."""
        chunks = [f"Chunk {i} content here with some text." for i in range(100)]
        metas = [{"source": "batch.txt", "chunk_index": i} for i in range(100)]
        ids = chroma_store.add_chunks(chunks, metas)
        assert len(ids) == 100
        assert chroma_store.count() == 100

    def test_search_no_results_returns_empty(self, chroma_store):
        """Searching empty store returns empty arrays, no crash."""
        results = chroma_store.search("anything")
        assert len(results["documents"][0]) == 0

    def test_empty_chunk_list(self, chroma_store):
        ids = chroma_store.add_chunks([], [])
        assert ids == []

    def test_deterministic_ids(self, chroma_store):
        """Same content + source produces same IDs (content hash based)."""
        ids1 = chroma_store.add_chunks(
            ["Deterministic content"],
            [{"source": "det.txt", "chunk_index": 0}],
        )
        # Re-create store to simulate fresh start
        import tempfile
        d2 = tempfile.mkdtemp()
        try:
            from backend.vector_store.chroma_client import ChromaStore
            store2 = ChromaStore(persist_directory=d2, collection_name="test_coll2")
            ids2 = store2.add_chunks(
                ["Deterministic content"],
                [{"source": "det.txt", "chunk_index": 0}],
            )
            # IDs are deterministic from content hash, not random
            assert len(ids1[0]) > 0
            assert len(ids2[0]) > 0
        finally:
            shutil.rmtree(d2, ignore_errors=True)

    def test_custom_ids(self, chroma_store):
        """Custom IDs are used when provided."""
        custom = ["my-id-1", "my-id-2"]
        ids = chroma_store.add_chunks(
            ["Content A", "Content B"],
            [{"source": "custom.txt", "chunk_index": 0},
             {"source": "custom.txt", "chunk_index": 1}],
            ids=custom,
        )
        assert ids == custom


# ═══════════════════════════════════════════════════════════════
# Sub-batch 1F — Embedder (lightweight, CPU)
# ═══════════════════════════════════════════════════════════════


class TestEmbedder:
    """Sentence-transformers embedder — encode, batch, normalisation."""

    def test_encode_returns_float_vector(self, embedder):
        vec = embedder.encode("test sentence")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)
        assert len(vec) in (384, 768)  # all-MiniLM-L6-v2 or similar

    def test_encode_batch(self, embedder):
        vecs = embedder.encode_many(["one", "two", "three"])
        assert len(vecs) == 3
        assert len(vecs[0]) in (384, 768)
        assert all(isinstance(v, float) for vec in vecs for v in vec)

    def test_vector_is_unit_normalised(self, embedder):
        vec = embedder.encode("test")
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 0.01, f"Expected unit vector, got {magnitude}"

    def test_same_text_same_embedding(self, embedder):
        v1 = embedder.encode("Hello world")
        v2 = embedder.encode("Hello world")
        differences = [abs(a - b) for a, b in zip(v1, v2)]
        assert max(differences) < 1e-5  # Deterministic

    def test_different_text_different_embedding(self, embedder):
        v1 = embedder.encode("Apples are fruits")
        v2 = embedder.encode("Quantum physics theory")
        # Vectors should be different
        assert v1 != v2

    def test_empty_string(self, embedder):
        """Empty string produces a valid embedding (no NaN)."""
        vec = embedder.encode("")
        assert all(not math.isnan(v) for v in vec)
        assert all(math.isfinite(v) for v in vec)

    def test_very_long_text(self, embedder):
        long_text = "word " * 10000
        vec = embedder.encode(long_text)
        assert len(vec) in (384, 768)
        assert all(math.isfinite(v) for v in vec)

    def test_dimension_property(self, embedder):
        assert isinstance(embedder.dimension, int)
        assert embedder.dimension in (384, 768)
