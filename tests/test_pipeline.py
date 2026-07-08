"""Batch 2 — Pipeline Tests.

Integration tests for the RAG pipeline with mocked LLM and real ChromaDB.
No Ollama required.

Sub-batches:
- 2A: RAGPipeline with MockLLMClient
- 2B: Reranker (if installed)
- 2C: Full pipeline path (chunk -> embed -> store -> retrieve)

Run:  pytest tests/test_pipeline.py -v
"""

import asyncio
import time
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.rag_pipeline import RAGPipeline
from backend.core.chunker import DocumentChunker
from backend.core.embedder import Embedder


# ═══════════════════════════════════════════════════════════════
# Sub-batch 2A -- RAGPipeline with MockLLMClient
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestRAGPipelineMocked:
    """RAGPipeline using mocked LLM and real ChromaStore."""

    def seed_chroma(self, chroma_store):
        chunks = [
            "The capital of France is Paris. The Eiffel Tower is in Paris.",
            "Python is a high-level programming language created by Guido van Rossum.",
            "The French Revolution began in 1789 and ended in 1799.",
            "Python supports object-oriented, functional, and procedural programming.",
        ]
        metas = [
            {"source": "geography.txt", "chunk_index": 0},
            {"source": "programming.txt", "chunk_index": 0},
            {"source": "history.txt", "chunk_index": 0},
            {"source": "programming.txt", "chunk_index": 1},
        ]
        chroma_store.add_chunks(chunks, metas)

    async def test_answer_returns_expected_keys(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("What is the capital of France?")
        assert "answer" in result
        assert "sources" in result
        assert "latency_ms" in result
        assert "query" in result

    async def test_answer_includes_sources(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("What is the capital of France?")
        sources = result["sources"]
        assert len(sources) > 0
        for s in sources:
            assert "source" in s
            assert "content_preview" in s
            assert "score" in s

    async def test_sources_limited_by_top_k(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("programming", top_k=2)
        assert len(result["sources"]) <= 2

    async def test_n_results_controls_initial_retrieval(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("France", n_results=100)
        assert len(result["sources"]) > 0

    async def test_where_filter_passed_to_vector_store(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer(
            "programming", where={"source": "programming.txt"}
        )
        sources = result["sources"]
        assert len(sources) > 0
        for s in sources:
            assert s["source"] == "programming.txt"

    async def test_score_is_rounded_to_4_decimal_places(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("France")
        for s in result["sources"]:
            score_str = str(s["score"])
            if "." in score_str:
                decimals = len(score_str.split(".")[1])
                assert decimals <= 4, f"Score {s['score']} has {decimals} decimals"

    async def test_empty_vector_store_does_not_crash(self, rag_pipeline):
        result = await rag_pipeline.answer("anything")
        assert "answer" in result
        assert len(result["sources"]) == 0

    async def test_latency_is_positive_integer(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("France")
        assert isinstance(result["latency_ms"], int)
        assert result["latency_ms"] >= 0

    async def test_streaming_returns_async_generator(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("France", stream=True)
        assert hasattr(result["answer"], "__aiter__")

    async def test_streaming_tokens_reassemble(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("France", stream=True)
        tokens = []
        async for token in result["answer"]:
            tokens.append(token)
        full = "".join(tokens)
        assert len(full) > 0

    async def test_sources_contain_content_preview(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("France")
        for s in result["sources"]:
            assert len(s["content_preview"]) > 0
            assert len(s["content_preview"]) <= 210

    async def test_rag_pipeline_uses_context_template(self, rag_pipeline, chroma_store, mock_llm):
        self.seed_chroma(chroma_store)
        await rag_pipeline.answer("France")
        assert "--- CONTEXT ---" in mock_llm.last_prompt
        assert "France" in mock_llm.last_prompt
        assert "--- END CONTEXT ---" in mock_llm.last_prompt

    async def test_rag_pipeline_calls_llm_once(self, rag_pipeline, chroma_store, mock_llm):
        self.seed_chroma(chroma_store)
        assert mock_llm.generated_count == 0
        await rag_pipeline.answer("France")
        assert mock_llm.generated_count == 1

    async def test_query_with_special_characters(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("SELECT * FROM documents; -- injection test")
        assert "answer" in result

    async def test_query_with_unicode(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("<<C3><BF><C3><A9>><C3><A0><C3><A8><C3><B2><C3><B9>?")
        assert "answer" in result

    async def test_very_long_query(self, rag_pipeline, chroma_store):
        long_q = "test " * 2000
        result = await rag_pipeline.answer(long_q)
        assert "answer" in result

    async def test_answer_returns_query_in_result(self, rag_pipeline, chroma_store):
        self.seed_chroma(chroma_store)
        result = await rag_pipeline.answer("What is Python?")
        assert result["query"] == "What is Python?"


# ═══════════════════════════════════════════════════════════════
# Sub-batch 2B -- Reranker
# ═══════════════════════════════════════════════════════════════


class TestReranker:
    """Reranker tests -- runs only if cross-encoder is installed."""

    @classmethod
    def setup_class(cls):
        try:
            from backend.core.reranker import Reranker
            cls.available = True
        except ImportError:
            cls.available = False

    def test_reranker_importable(self):
        from backend.core.reranker import Reranker
        assert Reranker is not None

    @pytest.mark.skipif("not TestReranker.available")
    def test_reranker_returns_sorted_results(self):
        from backend.core.reranker import Reranker
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        docs = [
            "Paris is the capital of France.",
            "Python is a programming language.",
            "The Eiffel Tower is a famous landmark.",
        ]
        results = reranker.rerank("What is the capital of France?", docs, top_k=2)
        assert len(results) == 2
        assert results[0]["relevance_score"] >= results[1]["relevance_score"]

    @pytest.mark.skipif("not TestReranker.available")
    def test_reranker_handles_empty_list(self):
        from backend.core.reranker import Reranker
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        results = reranker.rerank("test query", [], top_k=5)
        assert results == []

    @pytest.mark.skipif("not TestReranker.available")
    def test_reranker_fewer_docs_than_top_k(self):
        from backend.core.reranker import Reranker
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        docs = ["Only one document here."]
        results = reranker.rerank("test", docs, top_k=10)
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════
# Sub-batch 2C -- Full Pipeline Integration
# ═══════════════════════════════════════════════════════════════


class TestFullPipeline:
    """End-to-end: chunk -> embed -> store -> retrieve pipeline path."""

    def test_chunk_embed_store_retrieve_roundtrip(self, chroma_store, chunker):
        content = (
            "The Great Wall of China is a series of fortifications. "
            "It was built across the historical northern borders of China. "
            "The wall was built by multiple dynasties over centuries."
        )
        chunks, metadatas = chunker.chunk(content, source="china.txt")
        ids = chroma_store.add_chunks(chunks, metadatas)
        assert len(ids) == len(chunks)
        results = chroma_store.search("Great Wall of China", n_results=1)
        assert len(results["documents"][0]) >= 1
        assert "Great Wall" in results["documents"][0][0]

    def test_multiple_documents_isolated(self, chroma_store, chunker):
        doc1 = "Quantum mechanics describes nature at the smallest scales."
        doc2 = "Classical mechanics describes motion of macroscopic objects."
        chunks1, metas1 = chunker.chunk(doc1, source="quantum.txt")
        chunks2, metas2 = chunker.chunk(doc2, source="classical.txt")
        chroma_store.add_chunks(chunks1, metas1)
        chroma_store.add_chunks(chunks2, metas2)
        results = chroma_store.search("Planck scale atoms particles")
        top_source = results["metadatas"][0][0]["source"]
        assert "quantum" in top_source

    def test_update_reflects_new_content(self, chroma_store, chunker):
        content_old = "Old version: Paris is the capital."
        content_new = "New version: Rome is the capital of Italy."
        chunks_old, metas_old = chunker.chunk(content_old, source="capital.txt")
        chroma_store.add_chunks(chunks_old, metas_old)
        chroma_store.delete_document("capital.txt")
        chunks_new, metas_new = chunker.chunk(content_new, source="capital.txt")
        chroma_store.add_chunks(chunks_new, metas_new)
        results = chroma_store.search("capital of Italy")
        top = results["documents"][0][0]
        assert "Rome" in top
        assert "Old" not in top

    def test_semantic_search_vs_lexical(self, chroma_store, chunker):
        docs = [
            "The study of heredity and variation in organisms.",
            "The process by which species change over time.",
            "Financial accounting standards and reporting.",
        ]
        for i, doc in enumerate(docs):
            ch, me = chunker.chunk(doc, source=f"doc{i}.txt")
            chroma_store.add_chunks(ch, me)
        results = chroma_store.search("genetics")
        top_doc = results["documents"][0][0]
        assert "heredity" in top_doc or "species" in top_doc

    def test_stopwords_do_not_break_search(self, chroma_store, chunker):
        ch, me = chunker.chunk("The quick brown fox jumps over the lazy dog.", source="fox.txt")
        chroma_store.add_chunks(ch, me)
        results = chroma_store.search("the a an of in to is")
        assert len(results["documents"][0]) > 0
