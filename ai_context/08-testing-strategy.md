# Testing Strategy — AI RAG System

## Philosophy

This document defines an **intense, defence-in-depth testing strategy** for the AI RAG System. Every module, every edge case, every failure mode is covered. Tests are executed in **6 parallelisable batches** — each batch is a self-contained set of tests that can run independently.

### Test Pyramid

```
         ╱╲
        ╱  ╲          E2E (Batch 5) — full system, real Ollama + ChromaDB
       ╱    ╲         Integration (Batch 2, 3) — RAG pipeline, API endpoints
      ╱______╲
     ╱        ╲       Unit (Batch 1, 4) — config, PDS, vector store, LLM client
    ╱          ╲
   ╱____________╲     Property / Fuzz (Batch 6) — invariants, adversarial input
```

### Current Coverage Baseline (Final)

| Module | Lines | Tests | Status |
|--------|-------|-------|--------|
| `config.py` | 56 | 9 (foundation) | ✅ |
| `chunker.py` | 201 | 24 + 8 property (foundation) | ✅ |
| `embedder.py` | 47 | 7 (foundation) | ✅ |
| `llm_client.py` | 131 | 13 unit + 4 integration (llm) | ✅ |
| `rag_pipeline.py` | 141 | 18 (pipeline) | ✅ |
| `chroma_client.py` | 118 | 15 (foundation) | ✅ |
| `pds/models.py` | 140 | 16 (foundation) | ✅ |
| `pds/repository.py` | 205 | 16 (foundation) | ✅ |
| `pds/file_store.py` | 102 | 14 + 6 property (foundation) | ✅ |
| `routers/documents.py` | 167 | 19 (api) | ✅ (needs live server) |
| `routers/chat.py` | 215 | 17 (api) + 12 (e2e) | ✅ (needs live server) |
| `routers/admin.py` | 127 | 6 (api) + 3 (e2e) | ✅ (needs live server) |
| `reranker.py` | 51 | 4 (pipeline) | ✅ |

**Achieved: 183 non-live tests passing. All 6 batches implemented.**

---

## Batch Structure

| Batch | Focus | Files | Type | Ollama Required? |
|-------|-------|-------|------|------------------|
| **1** | Foundation | `test_foundation.py` | Unit | No |
| **2** | Pipeline | `test_pipeline.py` | Integration | No (mocked) |
| **3** | API | `test_api_intense.py` | Integration | Yes (live) |
| **4** | LLM | `test_llm.py` | Unit + Integration | No (mocked) |
| **5** | End-to-End | `test_e2e.py` | E2E | Yes (live) |
| **6** | Property/Fuzz | `test_property.py` | Property | No |

Each batch has its own runner and can be invoked independently. All batches share the harness in `tests/harness.py`.

---

## Batch 1 — Foundation (`test_foundation.py`)

**Goal:** Test every module that has zero external dependencies (no network, no Ollama, no ChromaDB).

### Sub-batch 1A — Config

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Default values match expected schema | Accidental default changes |
| 2 | `allowed_file_types` includes all extensions | Feature regression |
| 3 | `env_prefix` is correctly set to `RAG_` | Config loading breakage |
| 4 | Invalid `temperature` (>1.0 or <0.0) raises validation error | Pydantic validation gap |
| 5 | `max_file_size_mb` is positive integer | Misconfiguration |
| 6 | `chunk_overlap < chunk_size` invariant (enforced by code) | Silent logic bugs |

### Sub-batch 1B — PDS Models (ORM)

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | All 5 tables created via `Base.metadata.create_all` | Migration gaps |
| 2 | Document column set complete (id, filename, file_path, file_hash, uploaded_at, etc.) | Schema drift |
| 3 | UUID primary keys are unique and non-null | PK generation issues |
| 4 | ForeignKey cascade behavior (delete document cascades to chunks) | Orphan records |
| 5 | ChatSession ↔ ChatMessage relationship bidirectional | ORM relationship breakage |
| 6 | Default values set correctly (status="pending", category="general") | Constraint violations |
| 7 | IngestionRecord status enum constraint (must be pending/running/success/failed) | Data integrity |
| 8 | Timestamps are timezone-aware (UTC) | Timezone bugs |
| 9 | `sources_json` stores and retrieves list of dicts | JSON serialisation |
| 10 | Composite index on (document_id, chunk_index) enforced | Query performance |

### Sub-batch 1C — FileStore

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | `store()` creates date-nested directory | File organisation |
| 2 | `store()` returns absolute path | Path handling |
| 3 | `read_text()` reads back written content | Round-trip integrity |
| 4 | `read_bytes()` returns raw bytes | Binary file support |
| 5 | `md5()` matches `hashlib.md5` of file content | Hash consistency |
| 6 | `md5_bytes()` matches `hashlib.md5` of bytes | Hash consistency |
| 7 | `delete()` removes file and returns True | Cleanup correctness |
| 8 | `delete()` on non-existent returns True (idempotent) | Error handling |
| 9 | `usage()` returns correct counts after multiple stores | Stats accuracy |
| 10 | `read_text()` returns None for binary content | Graceful fallback |
| 11 | `read_text()` returns None for missing file | Error handling |
| 12 | Concurrent `store()` calls don't collide | Race condition |
| 13 | File name has unique prefix (uuid hex) | Collision resistance |
| 14 | `store()` with empty bytes writes empty file | Edge case |

### Sub-batch 1D — Chunker (all strategies)

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Recursive: splits on paragraph boundary first | Separation quality |
| 2 | Recursive: overlap preserves content continuity | Context window |
| 3 | Recursive: single short chunk | Boundary handling |
| 4 | Recursive: text exactly at chunk_size produces one chunk | Exact fit |
| 5 | Recursive: empty text returns empty list | Edge case |
| 6 | Recursive: assertion on overlap >= chunk_size | Invariant enforcement |
| 7 | Markdown: splits at `# `, `## `, `### ` heading boundaries | Structure awareness |
| 8 | Markdown: no headings falls back to recursive | Graceful degradation |
| 9 | Markdown: heading at exact chunk boundary | Boundary precision |
| 10 | Token: approximate token count matches 4:1 ratio | Approximation accuracy |
| 11 | Token: overlap in token space | Continuity |
| 12 | Token: very long text produces expected chunk count | Scaling |
| 13 | Sentence: respects `.!?` sentence boundaries | Precision |
| 14 | Sentence: max_chars threshold enforced | Size limit |
| 15 | Sentence: text with no sentence breaks falls through | Fallback behaviour |
| 16 | DocumentChunker: selects Recursive for `.txt` | Strategy routing |
| 17 | DocumentChunker: selects Markdown for `.md` / `.html` | Strategy routing |
| 18 | DocumentChunker: selects Token for `.py` / `.js` / `.ts` | Strategy routing |
| 19 | DocumentChunker: selects Sentence for `.csv` | Strategy routing |
| 20 | DocumentChunker: defaults to Recursive for unknown ext | Default behaviour |
| 21 | DocumentChunker: metadata includes source, chunk_index, total_chunks, chunk_size, strategy | Metadata completeness |
| 22 | DocumentChunker: chunk_index is sequentially correct | Ordering |
| 23 | DocumentChunker: total_chunks reflects actual count | Consistency |
| 24 | DocumentChunker: explicit `file_type` overrides extension | Override logic |

### Sub-batch 1E — ChromaStore (mocked / temp dir)

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | `add_chunks()` returns correct number of IDs | Ingestion |
| 2 | IDs auto-generated from content hash + source | Deterministic IDs |
| 3 | `add_chunks()` with custom IDs uses them | Custom ID support |
| 4 | `count()` reflects real count after adds | State tracking |
| 5 | `search()` returns expected structure (documents, metadatas, distances, ids) | API contract |
| 6 | `search()` with `n_results` limits correctly | Pagination |
| 7 | `search()` with `where` filter narrows results | Metadata filtering |
| 8 | `delete_document()` removes only chunks for that source | Targeted deletion |
| 9 | `delete_document()` returns deleted count | Idempotency |
| 10 | `reset_collection()` clears all data | Cleanup |
| 11 | `get_stats()` returns name and total_chunks | Observability |
| 12 | Batch ingestion (100+ chunks) doesn't exceed memory | Batch safety |
| 13 | Encodes same text deterministically | Embedding consistency |
| 14 | Empty chunk list returns 0 IDs | Edge case |
| 15 | Search with no results returns empty arrays | Graceful |

### Sub-batch 1F — Embedder (lightweight, CPU)

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | `encode()` returns list of floats with correct dimension (384 or 768) | Output contract |
| 2 | `encode_many()` returns correct batch count | Batch correctness |
| 3 | Encoded vector is unit-normalised (L2 norm ≈ 1.0) | Cosine readiness |
| 4 | Same text produces same embedding deterministically | Consistency |
| 5 | Empty string returns valid embedding (non-NaN) | Edge case |
| 6 | Very long text (10K chars) doesn't OOM | Memory safety |
| 7 | `dimension` property returns correct int | Introspection |

---

## Batch 2 — Pipeline (`test_pipeline.py`)

**Goal:** Integration tests for the RAG pipeline with mocked LLM and real ChromaDB (temp dir). No Ollama required.

### Sub-batch 2A — RAGPipeline with mocked LLM

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | `answer()` returns dict with answer, sources, latency_ms, query | Response contract |
| 2 | Answer includes sources from vector store context | RAG grounding |
| 3 | `where` filter is passed through to vector store search | Filter integrity |
| 4 | `top_k` truncates chunks before LLM | Truncation logic |
| 5 | `n_results` controls initial retrieval count | Search depth |
| 6 | Response latency is measured in milliseconds | Timing accuracy |
| 7 | `sources` in response contain source, content_preview, score | Source completeness |
| 8 | Score is rounded to 4 decimal places | Precision consistency |
| 9 | Empty vector store returns valid response with no sources | Graceful degradation |
| 10 | Query with special characters (SQL injection attempt) | Injection safety |
| 11 | Query with unicode/emoji characters | Unicode handling |
| 12 | Very long query (>5000 chars) is handled | Length boundary |
| 13 | Prompt template renders correctly with context and query | Template integrity |
| 14 | Streaming mode branches correctly | Code path coverage |

### Sub-batch 2B — Reranker (if installed)

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | `rerank()` returns top_k results sorted by relevance_score descending | Ordering |
| 2 | Reranker handles fewer docs than top_k gracefully | Boundary |
| 3 | Scores are between 0 and 1 | Score range |
| 4 | Empty document list returns empty list | Edge case |
| 5 | All documents identical — scores are equal but non-zero | Score behaviour |

### Sub-batch 2C — Pipeline Integration (Embedder + Chunker + Chroma + LLM mock)

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Chunk → Embed → Store → Retrieve round-trip works | Full path |
| 2 | Multiple documents indexed, query retrieves from correct one | Isolation |
| 3 | Query with stopwords retrieves relevant results | NLP robustness |
| 4 | Query semantically similar but lexically different | Semantic search |
| 5 | Document updated (delete + re-add) reflects new content | Freshness |

---

## Batch 3 — API (`test_api_intense.py`)

**Goal:** Test all API endpoints with live server. Requires Ollama running.

### Sub-batch 3A — Health & Admin

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | `GET /api/health` returns 200 with status="running" | Liveness |
| 2 | `GET /api/admin/health` returns component status | Deep health |
| 3 | `GET /api/admin/health` reports LLM availability | Dependency check |
| 4 | `GET /api/admin/stats` returns documents + vectors | Observability |
| 5 | `GET /api/admin/stats` returns uptime | Uptime tracking |
| 6 | `POST /api/admin/reindex` re-embeds all documents | Re-indexing |

### Sub-batch 3B — Document CRUD

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Upload `.txt` returns 200 with document_id + chunks | Basic upload |
| 2 | Upload `.md` with metadata (category, tags) | Metadata attachment |
| 3 | Upload `.pdf` parsed correctly | PDF parsing |
| 4 | Upload `.docx` parsed correctly | DOCX parsing |
| 5 | Upload duplicate content returns 409 (dedup) | Hash-based dedup |
| 6 | Upload file > max_file_size_mb returns 413 | Size enforcement |
| 7 | Upload unsupported file type returns 400 | Type validation |
| 8 | `GET /api/documents/` lists all documents | List endpoint |
| 9 | `GET /api/documents/` with category filter | Filtering |
| 10 | `GET /api/documents/` with file_type filter | Filtering |
| 11 | `GET /api/documents/` with pagination (limit, offset) | Pagination |
| 12 | `GET /api/documents/{id}` returns full document | Detail endpoint |
| 13 | `GET /api/documents/{id}` for non-existent returns 404 | Error handling |
| 14 | `DELETE /api/documents/{id}` removes file + chunks + PDS record | Full cleanup |
| 15 | `DELETE /api/documents/{id}` for non-existent returns 404 | Error handling |
| 16 | `GET /api/documents/stats` returns vector_chunks > 0 | Stats accuracy |
| 17 | Upload empty file returns 400 | Validation |
| 18 | Upload with invalid filename (path traversal: `../../etc`) | Security |
| 19 | Upload same filename twice — both stored (different IDs) | Collision handling |

### Sub-batch 3C — Chat & RAG

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | `POST /api/chat/sessions` creates and returns session_id | Session creation |
| 2 | `POST /api/chat/sessions` with custom title | Customisation |
| 3 | `GET /api/chat/sessions` returns recent sessions | Session list |
| 4 | `GET /api/chat/sessions?limit=5` respects limit | Pagination |
| 5 | `POST /api/chat/ask` returns answer + sources + latency | RAG response |
| 6 | Answer cites sources as `[Source: filename]` | Citation format |
| 7 | `POST /api/chat/ask` with existing session_id appends to history | Session continuity |
| 8 | `POST /api/chat/ask` with non-existent session returns 404 | Error handling |
| 9 | `POST /api/chat/ask` with document_filter scopes retrieval | Filtered search |
| 10 | `POST /api/chat/ask` with empty query returns 422 or graceful | Input validation |
| 11 | `POST /api/chat/ask/stream` returns SSE events | Streaming |
| 12 | SSE stream includes: sources → tokens → done | Event order |
| 13 | SSE stream tokens reassemble to a coherent answer | Stream integrity |
| 14 | `GET /api/chat/sessions/{id}/messages` returns history | Message history |
| 15 | Messages include role, content, sources, tokens_used, latency_ms | Message detail |
| 16 | Multiple questions in same session maintain context | Context window |
| 17 | Concurrent ask requests to same session don't corrupt | Thread safety |

### Sub-batch 3D — Error Handling

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Invalid JSON body returns 422 | Parsing errors |
| 2 | Missing required fields returns 422 | Validation |
| 3 | Wrong HTTP method returns 405 | Method not allowed |
| 4 | Request to uninitialised system returns 503 | Startup race |
| 5 | XSS in query string (`<script>alert(1)</script>`) | XSS prevention |
| 6 | Malformed UTF-8 in query | Encoding |
| 7 | Extremely long query (10K+ chars) truncates or handles | Length safety |

---

## Batch 4 — LLM Layer (`test_llm.py`)

**Goal:** Unit-test the `LLMClient` with mocked `httpx`, plus live integration tests. Both paths.

### Sub-batch 4A — Mocked LLMClient Unit Tests

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | `generate()` sends correct payload to Ollama API | Request contract |
| 2 | `generate()` with stream=True returns async generator | Streaming contract |
| 3 | `generate()` with stream=False returns full string | Sync contract |
| 4 | `chat()` sends messages array correctly | Chat contract |
| 5 | `chat()` with stream=True returns async generator | Streaming chat |
| 6 | `is_available()` returns True when model tag matches | Health check |
| 7 | `is_available()` returns False on HTTP error | Health check failure |
| 8 | `is_available()` returns False when model not in tag list | Model missing |
| 9 | `generate()` handles HTTP 4xx/5xx from Ollama | Upstream errors |
| 10 | `generate()` handles timeout gracefully | Network resilience |
| 11 | `generate()` handles malformed JSON in response stream | Response parsing |
| 12 | `generate()` with system_prompt includes it in payload | System prompt |
| 13 | `generate()` sets correct options (temperature, top_p, num_predict, keep_alive) | Options contract |
| 14 | `close()` disposes the underlying httpx client | Resource cleanup |
| 15 | Multiple `generate()` calls reuse same client connection | Connection reuse |

### Sub-batch 4B — Live Integration (requires Ollama)

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | `is_available()` returns True against real Ollama | Production check |
| 2 | `generate()` returns non-empty response | Real generation |
| 3 | `generate()` with stream=True yields tokens sequentially | Real streaming |
| 4 | `chat()` returns conversational response | Real chat |
| 5 | `generate()` respects temperature (low temp = deterministic) | Parameter effect |

---

## Batch 5 — End-to-End (`test_e2e.py`)

**Goal:** Full system tests that exercise the complete stack — upload → ingest → query → verify. Requires live Ollama + real ChromaDB.

### Sub-batch 5A — Basic Flow

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Upload .txt → verify chunked → ask question → verify answer contains expected content | Golden path |
| 2 | Upload .md → verify markdown-aware chunking | Strategy routing |
| 3 | Upload .pdf → verify text extraction | PDF parsing |
| 4 | Upload multiple docs → ask question → verify cross-document retrieval | Multi-doc |

### Sub-batch 5B — Session Workflow

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Create session → ask question → verify messages stored | Session lifecycle |
| 2 | Create session → ask 3 questions → verify all in history | History accumulation |
| 3 | Multiple sessions → verify isolation | Session isolation |
| 4 | Ask → delete → ask again → new session created | Auto-session |

### Sub-batch 5C — Streaming

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Stream request returns events → reconstruct answer | Stream round-trip |
| 2 | Stream answer matches non-stream answer (semantically) | Consistency |
| 3 | Multiple concurrent stream requests | Concurrency |

### Sub-batch 5D — Re-index

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Upload → ask (verify) → reindex → ask (verify still works) | Re-index integrity |
| 2 | Reindex with missing file reports status="file_missing" | Error reporting |

### Sub-batch 5E — Degraded / Error Recovery

| # | Test | What It Catches |
|---|------|-----------------|
| 1 | Stop Ollama → health reports degraded → restart → health recovers | Resilience |
| 2 | Upload corrupted PDF → server handles gracefully | Graceful error |
| 3 | Large concurrent uploads (10 files, 5MB each) | Stress |

---

## Batch 6 — Property-Based & Fuzz (`test_property.py`)

**Goal:** Use `hypothesis` (Python property-based testing library) to find edge cases no human would write.

### Sub-batch 6A — Chunker Invariants

| # | Property | Invariant |
|---|----------|-----------|
| 1 | Chunk concatenation | `"".join(chunks) == original` for recursive splitter with overlap=0 |
| 2 | Chunk size bound | `all(len(c) <= chunk_size + overlap)` for all chunks |
| 3 | Order preservation | Chunks appear in same order as original text |
| 4 | No data loss | Every character of original text appears in at least one chunk |
| 5 | Idempotent metadata | `total_chunks` in metadata matches `len(chunks)` |
| 6 | Strategy selection | File extensions map to correct strategies deterministically |
| 7 | Markdown heading preservation | Each markdown chunk starts with its heading |

### Sub-batch 6B — FileStore Invariants

| # | Property | Invariant |
|---|----------|-----------|
| 1 | Store → Read round-trip | `read_text(store(bytes, name)) == bytes.decode()` for text |
| 2 | Store → ReadBytes round-trip | `read_bytes(store(bytes, name)) == bytes` for any bytes |
| 3 | Delete idempotence | `delete(path)` called twice returns True then True |
| 4 | MD5 determinism | `md5_bytes(data)` always returns same hash for same data |
| 5 | MD5 collision resistance | Different data → different hashes (reasonable assumption) |
| 6 | Usage consistency | `usage()["total_size_bytes"] == sum(file sizes)` |
| 7 | Store isolation | Files with same name get different paths |
| 8 | Date directory creation | `store()` creates path containing today's YYYY-MM-DD |

### Sub-batch 6C — PDS Repository Invariants

| # | Property | Invariant |
|---|----------|-----------|
| 1 | Get after Add | `get_document(add_document(...).id)` returns the document |
| 2 | Delete idempotent | `delete_document(id)` first call True, second False |
| 3 | Stats consistency | `stats.total_documents == len(get_all_documents())` |
| 4 | Message ordering | `get_session_messages(id)` returns in chronological order |
| 5 | Session touch | After adding message, session `updated_at > created_at` |
| 6 | Non-existent IDs | `get_document(uuid())` returns None (not error) |

### Sub-batch 6D — ChromaStore Invariants

| # | Property | Invariant |
|---|----------|-----------|
| 1 | Add → Count | `count() == count + len(added)` |
| 2 | Delete reduces count | `count()` decreases by `delete_document(...)` return value |
| 3 | Search returns <= n_results | `len(search(q, n_results=n)) <= n` |
| 4 | Empty query returns results | `search("")` doesn't crash |
| 5 | Deterministic retrieval | Same query returns same top result deterministically |

### Sub-batch 6E — Fuzz Inputs (adversarial)

| # | Input Type | Example |
|---|------------|---------|
| 1 | Null bytes in text | `"\x00\x00\x00"` |
| 2 | SQL injection attempts | `"'; DROP TABLE documents; --"` |
| 3 | Unicode attacks | Right-to-left override, zero-width spaces, homoglyphs |
| 4 | Very deep nesting | `"A\n\nB\n\nC\n\n..."` repeated 10K times |
| 5 | Binary data as text | Random bytes, PNG bytes |
| 6 | JSON injection | `{"source": "<script>...</script>"}` |
| 7 | Path traversal | `"../../../etc/passwd"` as filename |
| 8 | Extremely long words | `"A" * 100000` |
| 9 | Regex ReDoS | `"(a|aa)+"` repeated |
| 10 | Emoji/Unicode | Mixed script, combining characters, ZWJ sequences |
| 11 | HTML/XML tags | `<tag attr="x">text</tag>` with nested weirdness |
| 12 | Large metadata payload | Metadata dict with 1000+ keys |

---

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio hypothesis httpx
```

### Run All Tests

```bash
# Batch 1 — Foundation (no external deps)
python -m pytest tests/test_foundation.py -v

# Batch 2 — Pipeline (mocked, no Ollama)
python -m pytest tests/test_pipeline.py -v

# Batch 3 — API (requires live server at localhost:8000)
# Start server first:
python -m uvicorn backend.main:app &
python -m pytest tests/test_api_intense.py -v

# Batch 4a — LLM Unit (mocked)
python -m pytest tests/test_llm.py -v -m "not integration"

# Batch 4b — LLM Integration (requires Ollama)
python -m pytest tests/test_llm.py -v -m "integration"

# Batch 5 — E2E (requires all services)
python -m pytest tests/test_e2e.py -v

# Batch 6 — Property/Fuzz (no external deps)
python -m pytest tests/test_property.py -v --hypothesis-show-statistics
```

### Parallel Execution

```bash
# Run batches 1, 4a, 6 in parallel (no external deps)
python -m pytest tests/test_foundation.py tests/test_llm.py tests/test_property.py \
  -v -n 3 --dist loadscope
```

---

## Continuous Integration

### GitHub Actions Workflow (`.github/workflows/test.yml`)

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  foundation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r backend/requirements.txt pytest pytest-asyncio hypothesis
      - run: python -m pytest tests/test_foundation.py tests/test_property.py -v

  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r backend/requirements.txt pytest pytest-asyncio
      - run: python -m pytest tests/test_pipeline.py -v

  e2e:
    runs-on: ubuntu-latest
    services:
      ollama:
        image: ollama/ollama:latest
        ports: ["11434:11434"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r backend/requirements.txt pytest httpx
      - run: ollama pull llama3.1:8b  # from service container
      - run: python -m pytest tests/test_e2e.py tests/test_api_intense.py -v
```

---

## Coverage Targets

| Batch | Module | Target Line Coverage |
|-------|--------|---------------------|
| 1 | `config.py` | 100% |
| 1 | `pds/models.py` | 95% |
| 1 | `pds/file_store.py` | 95% |
| 1 | `core/chunker.py` | 95% |
| 1 | `core/embedder.py` | 90% |
| 1 | `vector_store/chroma_client.py` | 90% |
| 2 | `core/rag_pipeline.py` | 90% |
| 2 | `core/reranker.py` | 90% |
| 3 | `routers/documents.py` | 90% |
| 3 | `routers/chat.py` | 85% |
| 3 | `routers/admin.py` | 85% |
| 4 | `core/llm_client.py` | 95% |
| 5 | All modules (E2E) | 100% critical paths |
| 6 | All modules (Property) | 100% invariants |

**Overall target: 90%+ line coverage, 100% critical-path coverage.**

---

## Test Harness

All batches share a common test harness at `tests/conftest.py`:

```python
"""Shared fixtures and utilities for all test batches."""
import pytest
import tempfile
import shutil
import os
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from backend.config import AppConfig
from backend.pds.repository import PDSRepository
from backend.pds.file_store import FileStore
from backend.vector_store.chroma_client import ChromaStore
from backend.core.llm_client import LLMClient
from backend.core.rag_pipeline import RAGPipeline
from backend.core.embedder import Embedder
from backend.core.chunker import DocumentChunker


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def file_store(temp_dir: str) -> FileStore:
    return FileStore(base_path=os.path.join(temp_dir, "docs"))


@pytest.fixture
def chroma_store(temp_dir: str) -> Generator[ChromaStore, None, None]:
    store = ChromaStore(
        persist_directory=os.path.join(temp_dir, "chroma"),
        collection_name="test_coll",
    )
    yield store


@pytest.fixture
def pds_repo(temp_dir: str) -> Generator[PDSRepository, None, None]:
    db_path = os.path.join(temp_dir, "test.db")
    repo = PDSRepository(db_path=db_path)
    import asyncio
    asyncio.get_event_loop().run_until_complete(repo.initialize())
    yield repo
    asyncio.get_event_loop().run_until_complete(repo.close())


@pytest.fixture
def chunker() -> DocumentChunker:
    return DocumentChunker(chunk_size=512, chunk_overlap=64)


@pytest.fixture
def embedder() -> Embedder:
    return Embedder()


class MockLLMClient:
    """A fully mocked LLM client that returns canned responses."""
    
    def __init__(self, response: str = "Mock response based on context."):
        self.response = response
        self.generated_count = 0
    
    async def generate(self, prompt: str, system_prompt: str = "", stream: bool = False):
        self.generated_count += 1
        if stream:
            async def token_gen():
                for word in self.response.split():
                    yield word + " "
            return token_gen()
        return self.response
    
    async def chat(self, messages: list[dict], stream: bool = False):
        return self.response
    
    async def is_available(self) -> bool:
        return True
    
    async def close(self):
        pass


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def rag_pipeline(chroma_store, mock_llm) -> RAGPipeline:
    return RAGPipeline(vector_store=chroma_store, llm_client=mock_llm)
```
