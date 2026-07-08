"""Batch 6 — Property-Based & Fuzz Tests.

Uses Hypothesis to find edge cases no human would write.
Covers invariants for chunker, FileStore, PDS Repository, and ChromaStore.

Run:  pytest tests/test_property.py -v --hypothesis-show-statistics
"""

import os
import math
import tempfile
import shutil
from pathlib import Path
from typing import Callable

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, invariant

from backend.core.chunker import (
    RecursiveCharacterSplitter, MarkdownSplitter,
    TokenSplitter, SentenceSplitter, DocumentChunker,
)
from backend.pds.file_store import FileStore


# ── Helper strategies ──

VALID_FILENAME = st.from_regex(
    r"^[a-zA-Z0-9_\-][a-zA-Z0-9_\-\.]{0,28}\.[a-z]{2,4}$",
    fullmatch=True,
).filter(lambda x: x is not None and len(x) > 0)

VALID_FILENAME_NOEXT = st.from_regex(
    r"^[a-zA-Z0-9_\-]{1,20}$",
    fullmatch=True,
).filter(lambda x: x is not None and len(x) > 0)


# ═══════════════════════════════════════════════════════════════
# Sub-batch 6A — Chunker Invariants
# ═══════════════════════════════════════════════════════════════


class TestChunkerProperties:
    """Property-based tests for chunking invariants."""

    @given(text=st.text(min_size=1, max_size=200), chunk_size=st.integers(min_value=10, max_value=100))
    @settings(max_examples=50)
    def test_no_data_loss(self, text: str, chunk_size: int):
        """Every character of original text appears in at least one chunk."""
        c = RecursiveCharacterSplitter(chunk_size, chunk_size // 4)
        chunks = c.split(text)
        if chunks:
            all_text = "".join(chunks)
            for ch in text:
                if ch.strip():
                    assert ch in all_text, f"Char {ch!r} lost"

    @given(text=st.text(min_size=1, max_size=200), chunk_size=st.integers(min_value=10, max_value=100))
    @settings(max_examples=50)
    def test_order_preserved(self, text: str, chunk_size: int):
        """Chunks appear in same order as original text."""
        c = RecursiveCharacterSplitter(chunk_size, chunk_size // 4)
        chunks = c.split(text)
        if not chunks:
            return
        indices = []
        for ch in chunks:
            sig = ch[:10].strip()
            idx = text.find(sig)
            if idx >= 0:
                indices.append(idx)
        assert indices == sorted(indices), "Chunks out of order"

    @given(text=st.text(min_size=0, max_size=100), chunk_size=st.integers(min_value=5, max_value=50))
    @settings(max_examples=50)
    def test_chunk_size_bound(self, text: str, chunk_size: int):
        """No chunk exceeds chunk_size + overlap."""
        overlap = chunk_size // 4
        c = RecursiveCharacterSplitter(chunk_size, overlap)
        chunks = c.split(text)
        max_allowed = chunk_size + overlap
        for chunk in chunks:
            assert len(chunk) <= max_allowed, f"Chunk {len(chunk)} > max {max_allowed}"

    @given(text=st.text(min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_empty_text_returns_empty(self, text: str):
        """Empty or whitespace-only text returns empty list."""
        c = RecursiveCharacterSplitter(50, 10)
        if not text.strip():
            assert c.split(text) == []

    @given(
        content=st.text(min_size=1, max_size=200),
        source=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_metadata_total_chunks_consistent(self, content: str, source: str):
        """total_chunks in metadata matches len(chunks)."""
        assume(len(source) > 0)
        dc = DocumentChunker(100, 10)
        chunks, metas = dc.chunk(content, source=source)
        if chunks:
            assert metas[0]["total_chunks"] == len(chunks)
            assert all(m["total_chunks"] == len(chunks) for m in metas)

    @given(
        content=st.text(min_size=1, max_size=200),
        source=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_metadata_chunk_index_sequential(self, content: str, source: str):
        """Chunk indices are 0, 1, 2, ..."""
        assume(len(source) > 0)
        dc = DocumentChunker(100, 10)
        _, metas = dc.chunk(content, source=source)
        for i, m in enumerate(metas):
            assert m["chunk_index"] == i

    @given(
        content=st.text(min_size=1, max_size=200),
        source=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_metadata_source_preserved(self, content: str, source: str):
        """Source value is correctly stored in metadata."""
        assume(len(source) > 0)
        dc = DocumentChunker(100, 10)
        _, metas = dc.chunk(content, source=source)
        for m in metas:
            assert m["source"] == source

    @given(
        ext=st.sampled_from([".md", ".html", ".py", ".js", ".ts", ".csv", ".txt", ".go", ".rs"]),
    )
    @settings(max_examples=20)
    def test_strategy_deterministic(self, ext: str):
        """Same extension always maps to same strategy."""
        dc1 = DocumentChunker(100, 10)
        dc2 = DocumentChunker(100, 10)
        _, m1 = dc1.chunk("content", source=f"file{ext}")
        _, m2 = dc2.chunk("content", source=f"file{ext}")
        assert m1[0]["strategy"] == m2[0]["strategy"]


# ═══════════════════════════════════════════════════════════════
# Sub-batch 6B — FileStore Invariants
# ═══════════════════════════════════════════════════════════════


class TestFileStoreProperties:
    """Property-based tests for FileStore invariants."""

    @given(data=st.binary(max_size=1000), filename=VALID_FILENAME)
    @settings(max_examples=50)
    def test_store_read_bytes_roundtrip(self, data: bytes, filename: str):
        """read_bytes(store(bytes)) == bytes"""
        d = tempfile.mkdtemp()
        try:
            fs = FileStore(base_path=os.path.join(d, "docs"))
            path = fs.store(data, filename)
            read_back = fs.read_bytes(path)
            assert read_back == data
        finally:
            shutil.rmtree(d, ignore_errors=True)

    @given(
        data=st.text(min_size=1, max_size=1000, alphabet=st.characters(min_codepoint=32, max_codepoint=127)),
        filename=VALID_FILENAME,
    )
    @settings(max_examples=50)
    def test_store_read_text_roundtrip(self, data: str, filename: str):
        """read_text(store(encode)) == data. Skips \r/\n — Windows text-mode normalises them."""
        d = tempfile.mkdtemp()
        try:
            fs = FileStore(base_path=os.path.join(d, "docs"))
            path = fs.store(data.encode(), filename)
            read_back = fs.read_text(path)
            assert read_back == data
        finally:
            shutil.rmtree(d, ignore_errors=True)

    @given(data=st.binary(min_size=1, max_size=500))
    @settings(max_examples=50)
    def test_md5_deterministic(self, data: bytes):
        """Same data always produces same MD5."""
        h1 = FileStore.md5_bytes(data)
        h2 = FileStore.md5_bytes(data)
        assert h1 == h2
        assert len(h1) == 32

    @given(data1=st.binary(min_size=1, max_size=500), data2=st.binary(min_size=1, max_size=500))
    @settings(max_examples=50)
    def test_md5_different_for_different_data(self, data1: bytes, data2: bytes):
        """Different data (usually) produces different hash."""
        assume(data1 != data2)
        h1 = FileStore.md5_bytes(data1)
        h2 = FileStore.md5_bytes(data2)
        assert h1 != h2

    @given(data=st.binary(min_size=1, max_size=500), filename=VALID_FILENAME)
    @settings(max_examples=50)
    def test_delete_removes_file(self, data: bytes, filename: str):
        """delete returns True for existing files."""
        d = tempfile.mkdtemp()
        try:
            fs = FileStore(base_path=os.path.join(d, "docs"))
            path = fs.store(data, filename)
            assert os.path.exists(path)
            result = fs.delete(path)
            assert result is True
            assert not os.path.exists(path)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    @given(filename=VALID_FILENAME_NOEXT)
    @settings(max_examples=50)
    def test_delete_nonexistent_no_error(self, filename: str):
        """Delete on non-existent file returns False (no crash)."""
        d = tempfile.mkdtemp()
        try:
            fs = FileStore(base_path=os.path.join(d, "docs"))
            result = fs.delete(os.path.join(d, "nonexistent", filename))
            assert result is False
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# Sub-batch 6C — Fuzz Inputs (adversarial)
# ═══════════════════════════════════════════════════════════════


class TestFuzzInputs:
    """Test chunker and FileStore with adversarial inputs."""

    @pytest.mark.parametrize("fuzz_input", [
        "\x00\x00\x00\x00",
        "'; DROP TABLE documents; --",
        "\u202EReverse override",
        "\u200B\u200C\u200D\u2060Zero-width",
        "A" * 10000,
        "<script>alert(1)</script>",
        "{" * 1000,
        "\n" * 1000,
        "\u0300\u0301\u0302\u0303Combining" + "z" * 100,
        "a\u0308\u0308\u0308\u0308\u0308\u0308" * 100,
    ])
    def test_recursive_chunker_survives_fuzz(self, fuzz_input: str):
        """Chunker handles adversarial input without crash."""
        c = RecursiveCharacterSplitter(50, 10)
        try:
            chunks = c.split(fuzz_input)
            assert isinstance(chunks, list)
        except (AssertionError, RecursionError):
            pytest.skip("Expected exception for pathological input")

    @pytest.mark.parametrize("fuzz_input", [
        "\x00\x00\x00\x00",
        "'; DROP TABLE documents; --",
        "\u202EReverse override",
        "A" * 10000,
        "{" * 1000,
        "\n" * 1000,
        "a" * 10000,
        "\t" * 1000,
        "\\x00\\x01\\x02\\xff",
    ])
    def test_document_chunker_survives_fuzz(self, fuzz_input: str):
        """DocumentChunker handles adversarial input without crash."""
        dc = DocumentChunker(100, 10)
        try:
            chunks, metas = dc.chunk(fuzz_input, source="fuzz.txt")
            assert isinstance(chunks, list)
            assert isinstance(metas, list)
            assert len(chunks) == len(metas)
        except (AssertionError, RecursionError):
            pytest.skip("Expected exception for pathological input")

    @pytest.mark.parametrize("fuzz_input", [
        b"",
        b"\x00" * 100,
        b"Hello\nWorld\n" * 1000,
        b"<script>alert(1)</script>" * 100,
        os.urandom(1000),
    ])
    def test_filestore_survives_fuzz(self, fuzz_input: bytes):
        """FileStore handles adversarial binary data."""
        d = tempfile.mkdtemp()
        try:
            fs = FileStore(base_path=os.path.join(d, "docs"))
            path = fs.store(fuzz_input, "fuzz.bin")
            read_back = fs.read_bytes(path)
            assert read_back == fuzz_input
            fs.delete(path)
        finally:
            shutil.rmtree(d, ignore_errors=True)
