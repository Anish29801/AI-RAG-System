# Tasks — AI RAG System Testing

## Batch 1 — Foundation (Unit tests — no external deps)
- [x] `tests/conftest.py` — Shared fixtures (temp_dir, file_store, chroma_store, pds_repo, mock_llm, chunker, embedder)
- [x] `tests/test_foundation.py` — Config, PDS Models (ORM), FileStore, Chunker (all strategies), ChromaStore (mocked), Embedder ✅ 106/106

## Batch 2 — Pipeline (Integration — mocked LLM + real ChromaDB)
- [x] `tests/test_pipeline.py` — RAGPipeline with MockLLMClient, reranker tests, chunk→embed→store→retrieve round-trip ✅ 26/26

## Batch 3 — API (Integration — requires live server)
- [x] `tests/test_api_intense.py` — Health, Admin, Document CRUD, Chat/RAG, Streaming, Error handling ✅ (24 tests, needs localhost:8000)

## Batch 4 — LLM (Unit + Integration — mocked httpx + live Ollama)
- [x] `tests/test_llm.py` — LLMClient unit tests (mocked httpx) ✅ 13/13, LLMClient integration (live) ✅ 4 tests

## Batch 5 — End-to-End (Full system — requires all services)
- [x] `tests/test_e2e.py` — Golden path, session workflow, streaming, re-index, degraded recovery ✅ (28 tests, needs live server + Ollama)

## Batch 6 — Property/Fuzz (Hypothesis — no external deps)
- [x] `tests/test_pipeline.py` — RAGPipeline with MockLLMClient, reranker tests, chunk→embed→store→retrieve round-trip ✅ 26/26
- [x] `tests/test_property.py` — Chunker invariants, FileStore invariants, PDS invariants, Chroma invariants, Fuzz inputs ✅ 38/38

## Deployment
- [ ] Push all test files to GitHub
- [ ] Deploy frontend via @Agni (Netlify)
