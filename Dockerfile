# ── Stage 1: Build React frontend ──
FROM node:20-alpine AS frontend-builder
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ── Stage 2: Python runtime ──
FROM python:3.12-slim

WORKDIR /app

# System deps for PyMuPDF, sentence-transformers, ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    mupdf-tools \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy deps first (layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY backend/ backend/
COPY --from=frontend-builder /build/dist/ frontend/dist/

# Data directory (volume mount)
RUN mkdir -p /app/data/documents /app/data/chroma_db

ENV PYTHONPATH=/app \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000 \
    RAG_OLLAMA_URL=http://ollama:11434

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
