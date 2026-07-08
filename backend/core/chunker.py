"""Document chunking strategies for RAG ingestion.

Supports:
  - RecursiveCharacterSplit  — default, works for most text
  - MarkdownSplit            — heading-aware for .md / .html
  - TokenSplit               — token-count-aware for code
  - SentenceSplit            — sentence-boundary for precise retrieval

Each strategy returns (chunks: list[str], metadatas: list[dict]).
"""

import os
import re
from typing import Optional


# ── Recursive Character Splitter ──


class RecursiveCharacterSplitter:
    """Splits text at natural boundaries with configurable chunk size/overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        assert chunk_overlap < chunk_size, "overlap must be smaller than chunk_size"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        final = []
        separator = separators[0] if separators else ""

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = separator.join(good_splits)
                    if merged:
                        final.extend(self._merge(merged))
                    good_splits = []
                if separators:
                    final.extend(self._split(s, separators[1:]))
                else:
                    final.extend(self._merge(s))

        if good_splits:
            merged = separator.join(good_splits)
            if merged:
                final.extend(self._merge(merged))

        return final

    def _merge(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        return [c for c in chunks if c.strip()]


# ── Markdown Splitter ──


class MarkdownSplitter:
    """Splits markdown documents by heading boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._fallback = RecursiveCharacterSplitter(chunk_size, chunk_overlap)

    def split(self, text: str) -> list[str]:
        sections = re.split(r"\n(?=#{1,6}\s)", text)
        if len(sections) <= 1:
            return self._fallback.split(text)

        chunks = []
        for sec in sections:
            if len(sec) > self._fallback.chunk_size:
                chunks.extend(self._fallback.split(sec))
            else:
                chunks.append(sec)
        return [c for c in chunks if c.strip()]


# ── Token Splitter ──


class TokenSplitter:
    """Splits text by approximate token count (4 chars ≈ 1 token)."""

    def __init__(self, chunk_tokens: int = 256, overlap_tokens: int = 32):
        self.chunk_size = chunk_tokens * 4
        self.overlap = overlap_tokens * 4
        self._fallback = RecursiveCharacterSplitter(self.chunk_size, self.overlap)

    def split(self, text: str) -> list[str]:
        return self._fallback.split(text)


# ── Sentence Splitter ──


class SentenceSplitter:
    """Splits text at sentence boundaries."""

    def __init__(self, max_chars: int = 1024):
        self.max_chars = max_chars

    def split(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current = []
        for s in sentences:
            current.append(s)
            if len(" ".join(current)) >= self.max_chars:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks


# ── Chunker Factory ──


class DocumentChunker:
    """Selects chunking strategy based on file extension."""

    STRATEGIES = {
        ".md": "markdown",
        ".html": "markdown",
        ".htm": "markdown",
        ".py": "token",
        ".js": "token",
        ".ts": "token",
        ".go": "token",
        ".rs": "token",
        ".java": "token",
        ".csv": "sentence",
        ".txt": "recursive",
    }

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        default_strategy: str = "recursive",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_strategy = default_strategy

        self._splitters = {
            "recursive": RecursiveCharacterSplitter(chunk_size, chunk_overlap),
            "markdown": MarkdownSplitter(chunk_size, chunk_overlap),
            "token": TokenSplitter(chunk_size // 4, chunk_overlap // 4),
            "sentence": SentenceSplitter(chunk_size * 2),
        }

    def chunk(
        self,
        content: str,
        source: str = "",
        file_type: Optional[str] = None,
    ) -> tuple[list[str], list[dict]]:
        """Chunk content and return (chunks, metadatas)."""
        if file_type is None:
            _, ext = os.path.splitext(source)
            file_type = self.STRATEGIES.get(ext, self.default_strategy)

        splitter = self._splitters.get(file_type, self._splitters[self.default_strategy])
        chunks = splitter.split(content)

        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "strategy": file_type,
            })

        return chunks, metadatas
