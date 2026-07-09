# AI RAG System

A **fully free, local-first** AI system combining LLM, RAG, PDS, and Vector Database — no API costs, no data leaving your machine.

## Architecture

```
User → FastAPI → RAG Pipeline → ChromaDB (vectors)
                    ↓
                Ollama (LLM)
                    ↓
              Response + Citations
```

## Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| LLM | Ollama (Llama 3.1 / Mistral / Gemma) | Free |
| Vector DB | ChromaDB | Free |
| Embeddings | sentence-transformers | Free |
| PDS | SQLite + File System | Free |
| Backend | FastAPI + Python 3.11+ | Free |

## Quick Start

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# 2. Set up Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# 3. Start the server
uvicorn backend.main:app --reload --port 8000

# 4. Upload a document
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@document.txt"

# 5. Ask a question
curl -X POST http://localhost:8000/api/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about X?"}'
```

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration
│   ├── routers/             # API endpoints
│   ├── core/                # RAG pipeline logic
│   ├── pds/                 # Personal Data Store
│   └── vector_store/        # ChromaDB wrapper
├── docs/                    # 7 detailed design docs
│   ├── 01-architecture-overview.md
│   ├── 02-llm-layer.md
│   ├── 03-vector-database.md
│   ├── 04-rag-pipeline.md
│   ├── 05-pds-layer.md
│   ├── 06-api-and-backend.md
│   └── 07-deployment-and-free-tier.md
├── data/
│   ├── documents/           # Uploaded files
│   └── chroma_db/           # Vector database
├── .env                     # Configuration
└── README.md
```

## Documentation

See the `docs/` directory for 7 in-depth design documents covering architecture, LLM layer, vector database, RAG pipeline, PDS, API, and deployment.

## License

MIT
