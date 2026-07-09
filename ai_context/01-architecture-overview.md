# 01 — System Architecture Overview

> **A fully free, open-source RAG AI system using LLM + PDS + Vector Database**

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                       │
│              (CLI / Web UI / API Client)                    │
└──────────┬──────────────────────────────────────┬───────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────┐            ┌──────────────────────────┐
│                      │            │                          │
│   FASTAPI BACKEND    │◄──────────►│   PERSONAL DATA STORE    │
│   (API Layer)        │            │   (SQLite + File System) │
│                      │            │                          │
└──────┬───────┬───────┘            └──────────────────────────┘
       │       │
       ▼       ▼
┌──────────┐ ┌────────────────────┐
│  LLM     │ │  VECTOR DATABASE   │
│  Layer   │ │  (ChromaDB)        │
│ (Ollama) │ │                    │
│          │ │  ┌──────────────┐  │
│ Llama 3  │ │  │ Embedding    │  │
│ Mistral  │ │  │ Model (FREE) │  │
│ Gemma    │ │  └──────────────┘  │
└──────────┘ └────────────────────┘
```

### Core Components

| Component | Technology | License | Cost |
|-----------|-----------|---------|------|
| **LLM** | Ollama (Llama 3 / Mistral / Gemma) | MIT / Apache 2.0 | Free (local) |
| **Vector DB** | ChromaDB | Apache 2.0 | Free (local) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Apache 2.0 | Free |
| **Backend** | FastAPI + Python 3.11+ | MIT | Free |
| **PDS** | SQLite + local filesystem | Public Domain | Free |
| **UI** | Gradio (optional) OR Streamlit OR CLI | Apache 2.0 | Free |
| **RAG Pipeline** | LangChain / LlamaIndex / Custom | MIT | Free |

---

## 2. Key Architecture Decisions

### Decision 1: Local-First (Zero API Costs)
- Run everything on local hardware using Ollama for LLM inference
- No recurring API bills — the system works fully offline once models are pulled
- Embedding generation uses CPU or GPU locally

### Decision 2: ChromaDB over Pinecone/Weaviate Cloud
- ChromaDB is Apache 2.0 licensed, runs embedded in your Python process
- No rate limits, no size caps, no data leaving your machine
- Supports in-memory and persistent (on-disk) modes
- Full metadata filtering, MMR search, collection management

### Decision 3: Modular RAG Pipeline
- Document ingestion is decoupled from query handling
- Chunking strategies are configurable per document type
- Reranking step is optional (can be toggled off for speed)
- LLM provider can be swapped without changing retrieval logic

### Decision 4: SQLite as PDS Backend
- Zero-config, serverless, battle-tested
- Handles metadata, document registry, conversation history
- Files stored on disk; SQLite tracks paths, hashes, timestamps
- Entire PDS is a single `.db` file — easy to back up, sync, or migrate

---

## 3. Data Flow

### Ingestion Pipeline

```
Source Document
     │
     ▼
File Validation ──► Metadata Extraction ──► Chunking
                                               │
                                               ▼
                                   Embedding Generation
                                      (CPU/GPU)
                                               │
                                     ┌─────────┴────────┐
                                     ▼                  ▼
                              ChromaDB            PDS (SQLite)
                           (vector storage)    (file registry,
                                                metadata store)
```

### Query Pipeline

```
User Query
     │
     ▼
Query Transformation ──► Embed Query ──► Vector Search (ChromaDB)
     │                                      │
     │                                      ▼
     │                              Retrieve Top-K Chunks
     │                                      │
     │                                      ▼
     │                              Reranking (optional)
     │                                      │
     └─────────────────► Context Assembly ◄─┘
                              │
                              ▼
                      LLM Generation
                      (Ollama)
                              │
                              ▼
                     Response + Citations
```

---

## 4. Directory Structure

```
ai-rag-system/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration (env vars, defaults)
│   ├── routers/
│   │   ├── documents.py     # Upload, delete, list documents
│   │   ├── chat.py          # Conversation endpoints
│   │   └── admin.py         # System health, stats
│   ├── core/
│   │   ├── llm_client.py    # Ollama/FastAPI client abstraction
│   │   ├── rag_pipeline.py  # Orchestrates retrieval + generation
│   │   ├── chunker.py       # Document chunking strategies
│   │   ├── embedder.py      # Embedding generation
│   │   └── reranker.py      # Cross-encoder reranking
│   ├── pds/
│   │   ├── models.py        # SQLAlchemy/Pydantic models
│   │   ├── repository.py    # CRUD operations
│   │   └── file_store.py    # File I/O management
│   ├── vector_store/
│   │   └── chroma_client.py # ChromaDB wrapper
│   └── requirements.txt
├── ollama/                   # Ollama configuration notes
├── data/
│   ├── documents/           # Uploaded file storage
│   ├── chroma_db/           # ChromaDB persistent data
│   └── pds.db               # SQLite database
├── docs/                    # System documentation
│   ├── 01-architecture-overview.md
│   ├── 02-llm-layer.md
│   ├── 03-vector-database.md
│   ├── 04-rag-pipeline.md
│   ├── 05-pds-layer.md
│   ├── 06-api-and-backend.md
│   └── 07-deployment-and-free-tier.md
└── README.md
```

---

## 5. Technology Stack — Justification

| Need | Choice | Why |
|------|--------|-----|
| **LLM Inference** | Ollama | Simplest local LLM runner; supports all major open models; GPU/CPU auto-detect; REST API built-in |
| **LLM Models** | Llama 3.1 8B | Best quality-to-size ratio for local; 8K context; Apache 2.0 |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim, 80MB, runs on CPU in <50ms, good enough for semantic search |
| **Vector DB** | ChromaDB | Free, embedded, persistent, metadata filters, MMR, no server process |
| **Chunking** | RecursiveCharacterTextSplitter | Handles markdown, code, plain text; configurable overlap |
| **Reranking** | BAAI/bge-reranker-v2-m3 | Cross-encoder; ~500ms per query; significantly boosts precision |
| **Backend** | FastAPI | Async, auto-docs, Pydantic validation, high performance |
| **PDS** | SQLite + File System | Zero-dependency, atomic transactions, portable |
| **ORM** | SQLAlchemy (async) | Mature, well-typed, migration support |
| **Config** | Pydantic Settings | Type-safe config loading from env/.env |

---

## 6. Trade-offs and Constraints

| Constraint | Impact | Mitigation |
|-----------|--------|-----------|
| Local LLM is slower than GPT-4 | ~20-40 tok/s vs 100+ | Streaming responses; batch processing for ingestion |
| 8B model has less reasoning depth | Struggles with complex multi-hop queries | RAG provides external context; can swap to Qwen 32B or Mixtral if RAM permits |
| ChromaDB is single-node | No distributed scaling | Sufficient for personal/team use; data is portable to Qdrant if needed |
| CPU embedding is slower | ~200ms per chunk vs 50ms GPU | Batch embeddings; progress tracking |
| Context window limited | 8K tokens (Llama 3) vs 128K (GPT-4) | Efficient chunking; summarization of large contexts |

---

## 7. Evolution Path

```
Phase 1 (MVP)        Phase 2              Phase 3
┌──────────┐        ┌──────────┐         ┌──────────┐
│ Local    │──────► │ Multi-   │────────►│ Cloud    │
│ Single   │        │ User     │         │ Hybrid   │
│ Ollama   │        │ Auth     │         │ + GPU    │
│ ChromaDB │        │ Web UI   │         │ Replica  │
│ CLI/API  │        │ Feedback │         │ Scaling  │
└──────────┘        └──────────┘         └──────────┘
```

This document defines the "what" and "why" of the architecture. Each subsequent document dives deep into a single component.
