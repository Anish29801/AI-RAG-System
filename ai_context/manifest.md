# Manifest — Installed Applications & Packages

> Auto-updated log of all software installed for the AI RAG System.

---

## System Applications

| Application | Version | Install Method | Date | Status |
|------------|---------|---------------|------|--------|
| **Ollama** | 0.31.1 | `winget install --id Ollama.Ollama` | 2026-07-08 | ✅ |
| **Python** | 3.11.x | Pre-installed (`C:\Users\Anish\AppData\Local\Programs\Python\Python311\`) | — | ✅ |
| **Python** | 3.14.3 | Pre-installed (`C:\Python314\`) | — | ⏸️ (not used) |

---

## Python Virtual Environment

| Item | Value |
|------|-------|
| **Path** | `C:\Users\Anish\Desktop\AI\.venv\` |
| **Python Version** | 3.11.11 |
| **Created** | 2026-07-08 |

### Installed Packages (from `backend/requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| **fastapi** | 0.115.0 | Async API framework |
| **uvicorn** | 0.30.0 | ASGI server |
| **pydantic** | 2.9.0 | Data validation |
| **pydantic-settings** | 2.5.0 | Env-based config |
| **python-multipart** | 0.0.9 | File upload handling |
| **httpx** | 0.27.0 | Async HTTP client (Ollama API) |
| **aiosqlite** | 0.20.0 | Async SQLite driver |
| **sqlalchemy** | 2.0.35 | ORM for PDS |
| **sse-starlette** | 2.1.0 | Server-Sent Events (streaming) |
| **sentence-transformers** | 5.6.0 | Free embedding models |
| **chromadb** | 1.5.9 | Vector database |
| **python-docx** | 1.1.2 | DOCX file parsing |
| **PyMuPDF** | 1.24.0 | PDF text extraction |
| **psutil** | 6.0.0 | System monitoring |

### Transitive Dependencies (auto-installed)

| Package | Version | Required By |
|---------|---------|-------------|
| torch | 2.12.1 | sentence-transformers |
| transformers | 4.57.6 | sentence-transformers |
| numpy | 2.4.6 | sentence-transformers, chromadb |
| scikit-learn | 1.9.0 | sentence-transformers |
| scipy | 1.17.1 | sentence-transformers |
| huggingface-hub | 0.36.2 | sentence-transformers |
| tokenizers | 0.22.2 | chromadb, transformers |
| onnxruntime | 1.27.0 | chromadb |
| grpcio | 1.81.1 | chromadb |
| opentelemetry-* | 1.43.0 / 0.64b0 | chromadb |
| kubernetes | 36.0.2 | chromadb |
| aiohttp | 3.14.1 | kubernetes |
| Pillow | 12.3.0 | sentence-transformers |
| tqdm | 4.68.4 | sentence-transformers |
| PyYAML | 6.0.3 | chromadb |
| rich | 15.0.0 | typer (chromadb) |
| typer | 0.26.8 | chromadb |
| orjson | 3.11.9 | chromadb |
| bcrypt | 5.0.0 | chromadb |
| wrapt | 2.2.2 | opentelemetry |
| regex | 2026.6.28 | transformers |
| safetensors | 0.8.0 | transformers |
| joblib | 1.5.3 | scikit-learn |
| threadpoolctl | 3.6.0 | scikit-learn |
| networkx | 3.6.1 | torch |
| sympy | 1.14.0 | torch |
| jinja2 | 3.1.6 | torch |
| filelock | 3.29.7 | huggingface-hub, torch |
| fsspec | 2026.6.0 | huggingface-hub |
| psutil | 6.0.0 | system monitoring |
| certifi | 2026.6.17 | httpx, requests |
| urllib3 | 2.7.0 | requests |
| idna | 3.18 | httpx, requests |
| charset-normalizer | 3.4.9 | requests |
| pygments | 2.20.0 | rich |
| markdown-it-py | 4.2.0 | rich |
| mdurl | 0.1.2 | markdown-it-py |
| greenlet | 3.5.3 | sqlalchemy |
| anyio | 4.14.1 | httpx, starlette |
| h11 | 0.16.0 | uvicorn |
| click | 8.4.2 | uvicorn |
| colorama | 0.4.6 | uvicorn, tqdm |
| httptools | 0.8.0 | uvicorn |
| watchfiles | 1.2.0 | uvicorn |
| websockets | 16.0 | uvicorn |
| starlette | 0.38.6 | fastapi |
| typing-extensions | 4.16.0 | pydantic, fastapi |
| annotated-types | 0.7.0 | pydantic |
| pydantic-core | 2.23.2 | pydantic |
| tzdata | 2026.2 | pydantic |
| python-dotenv | 1.2.2 | pydantic-settings |
| sniffio | 1.3.1 | anyio, httpx |
| packaging | 26.2 | build |
| pyproject-hooks | 1.2.0 | build |
| setuptools | 65.5.0 | base (pre-installed) |
| pip | 24.0 | base (pre-installed) |
| six | 1.17.0 | python-dateutil, kubernetes |
| python-dateutil | 2.9.0.post0 | kubernetes |
| oauthlib | 3.3.1 | requests-oauthlib |
| requests-oauthlib | 2.0.0 | kubernetes |
| websocket-client | 1.9.0 | kubernetes |
| importlib-resources | 7.1.0 | chromadb |
| overrides | 7.7.0 | chromadb |
| tenacity | 9.1.4 | chromadb |
| mmh3 | 5.2.1 | chromadb |
| chroma-hnswlib | 0.7.3 | chromadb |
| posthog | 7.22.0 | chromadb (telemetry) |
| backoff | 2.2.1 | posthog |
| distro | 1.9.0 | posthog |
| pypika | 0.51.1 | chromadb |
| chromadb | 0.5.0 | vector database |
| lxml | 6.1.1 | python-docx |
| PyMuPDFb | 1.24.0 | PyMuPDF |
| flatbuffers | 25.12.19 | onnxruntime |
| protobuf | 7.35.1 | onnxruntime, opentelemetry |
| googleapis-common-protos | 1.75.0 | opentelemetry |
| propcache | 0.5.2 | aiohttp |
| multidict | 6.7.1 | aiohttp |
| aiosignal | 1.4.0 | aiohttp |
| frozenlist | 1.8.0 | aiohttp |
| aiohappyeyeballs | 2.7.1 | aiohttp |
| attrs | 26.1.0 | aiohttp |
| yarl | 1.24.2 | aiohttp |
| narwhals | 2.23.0 | scikit-learn |
| durationpy | 0.10 | kubernetes |
| asgiref | 3.11.1 | opentelemetry-instrumentation-asgi |
| shellingham | 1.5.4 | typer |
| annotated-doc | 0.0.4 | typer |
| mpmath | 1.3.0 | sympy |

---

## LLM Models

| Model | Size | Status | Date |
|-------|------|--------|------|
| **llama3.1:8b** | 4.9 GB | ✅ Pulled | 2026-07-08 |

---

## Configuration

| File | Status |
|------|--------|
| `.env` | ✅ Created |
| `backend/requirements.txt` | ✅ Created |

---

*Last updated: 2026-07-08*
