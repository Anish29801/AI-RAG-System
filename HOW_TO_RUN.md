# How to Run — AI RAG System

## Prerequisites

- Python 3.12+
- Ollama (with a model pulled, e.g. `llama3.1:8b`)
- Node.js 20+ (for frontend development)
- Docker (optional, for containerised deployment)

---

## Quick Start (Local)

### 1. Start Ollama

```bash
ollama pull llama3.1:8b
ollama serve
```

### 2. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000/api/health` to verify it's running.

### 4. Build and serve the frontend

**Production mode** (single server):

```bash
cd frontend && npm install && npm run build
# Frontend is now served by FastAPI at http://localhost:8000
```

**Development mode** (hot reload, two terminals):

```bash
# Terminal 1 — Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend dev server
cd frontend && npm install && npm run dev
# Opens http://localhost:5173 — proxies /api to backend
```

---

## Docker (Production)

### Start everything

```bash
docker compose up -d

# Pull the LLM model inside the container
docker exec ai-rag-ollama ollama pull llama3.1:8b
```

Open `http://localhost:8000`.

### Stop

```bash
docker compose down
```

To also delete persisted data (documents, chat history, vectors):

```bash
docker compose down -v
```

---

## Running Tests

### All tests (requires live server + Ollama)

```bash
# Terminal 1 — Start the server
uvicorn backend.main:app --port 8000

# Terminal 2 — Run all tests
python -m pytest tests/ -v
```

### Only offline tests (no services needed)

```bash
python -m pytest tests/test_foundation.py tests/test_pipeline.py tests/test_property.py tests/test_llm.py -m "not integration" -v
```

### Test breakdown

| Batch | Command | Requires | Count |
|-------|---------|----------|-------|
| Foundation | `pytest tests/test_foundation.py` | Nothing | 106 |
| Pipeline | `pytest tests/test_pipeline.py` | Nothing | 26 |
| API | `pytest tests/test_api_intense.py` | Server :8000 | 28 |
| LLM unit | `pytest tests/test_llm.py -m "not integration"` | Nothing | 13 |
| LLM live | `pytest tests/test_llm.py -m integration` | Ollama | 4 |
| E2E | `pytest tests/test_e2e.py` | Server + Ollama | 22 |
| Property | `pytest tests/test_property.py` | Nothing | 38 |

---

## Configuration

All settings are controlled via environment variables with the `RAG_` prefix or a `.env` file in the project root.

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `RAG_LLM_MODEL` | `llama3.1:8b` | LLM model name |
| `RAG_HOST` | `0.0.0.0` | Server bind address |
| `RAG_PORT` | `8000` | Server port |
| `RAG_DEBUG` | `true` | Enable debug mode (disables rate limiting) |
| `RAG_CHUNK_SIZE` | `512` | Document chunk size (characters) |
| `RAG_CHUNK_OVERLAP` | `64` | Chunk overlap |
| `RAG_TOP_K` | `5` | Documents to retrieve per query |
| `RAG_MAX_FILE_SIZE_MB` | `50` | Max upload file size |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |

---

## Project Structure

```
.
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration (env vars + .env)
│   ├── core/                   # RAG pipeline, embedder, chunker, LLM client
│   ├── pds/                    # Personal Data Store (SQLite + file system)
│   ├── routers/                # API endpoints (documents, chat, admin)
│   ├── vector_store/           # ChromaDB wrapper
│   └── middleware/             # Rate limiting, logging
├── frontend/
│   ├── src/                    # React source
│   ├── dist/                   # Production build output
│   ├── package.json
│   └── vite.config.js
├── tests/                      # 237 tests across 6 batches
├── data/                       # Runtime data (gitignored)
├── Dockerfile
├── docker-compose.yml
└── HOW_TO_RUN.md
```

## Troubleshooting

**"Address already in use" on port 8000**

```bash
# Find and kill the process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**"Connection refused" on Ollama**

Ensure Ollama is running: `ollama serve`

**Frontend shows blank page**

Make sure you've run `npm run build` in `frontend/` or use the dev server.
