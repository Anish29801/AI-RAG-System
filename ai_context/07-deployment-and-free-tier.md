# 07 — Deployment & Free Tier Strategy

> **Deploy the entire AI RAG system for $0 — local, cloud, and hybrid strategies**

---

## 1. Deployment Philosophy

The system is designed with **local-first** as the primary deployment mode. This eliminates:

- API costs ($0/month inference)
- Data privacy concerns (everything runs on your machine)
- Rate limits (unlimited queries)
- Internet dependency (works fully offline)

**But** — if your hardware is limited, we provide free cloud fallbacks.

---

## 2. Deployment Matrix

| Scenario | Hardware | Best For | Total Cost |
|----------|----------|----------|------------|
| **Local (CPU)** | 8GB+ RAM, no GPU | Development, light use | **$0** |
| **Local (GPU)** | 8GB+ VRAM (NVIDIA) | Production personal use | **$0** |
| **Local + Cloud LLM** | Any RAM + Groq free API | Fast responses, limited budget | **$0** |
| **Cloud VM (CPU)** | Oracle Cloud free tier | 24/7 server | **$0** |
| **Cloud VM (GPU)** | $0.50/hr spot instance | Heavy batch processing | ~$36/month |
| **Hybrid** | Local ChromaDB + Cloud LLM | Balanced | **$0** |

---

## 3. Local Deployment (Primary)

### Hardware Requirements

| Component | Minimum | Recommended | Ideal |
|-----------|---------|-------------|-------|
| **RAM** | 8GB | 16GB | 32GB |
| **Storage** | 10GB free | 50GB SSD | 100GB SSD |
| **GPU** | None (CPU only) | NVIDIA 8GB VRAM | NVIDIA 16GB+ VRAM |
| **CPU** | 4 cores | 8 cores | 12+ cores |
| **OS** | Windows/Linux/macOS | Linux (Ubuntu 22.04) | Linux + WSL2 |

### Step-by-Step Local Deployment

#### Step 1: Install Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from ollama.com or use WSL2
wsl --install -d Ubuntu
# Inside WSL:
curl -fsSL https://ollama.com/install.sh | sh
```

#### Step 2: Pull Models

```bash
# Primary model for RAG
ollama pull llama3.1:8b

# Lightweight model for testing
ollama pull llama3.2:3b

# Embedding model (alternative to sentence-transformers)
ollama pull nomic-embed-text

# Verify
ollama list
```

#### Step 3: Keep Ollama Running

```bash
# Start Ollama service
ollama serve &

# Or as a systemd service
sudo systemctl start ollama
sudo systemctl enable ollama  # Auto-start on boot

# Pre-load model (avoids cold start delay)
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "llama3.1:8b", "prompt": "ping", "keep_alive": -1}'
```

#### Step 4: Set Up Python Environment

```bash
# Create project directory
mkdir ~/ai-rag-system && cd ~/ai-rag-system

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend/requirements.txt
```

#### Step 5: Configure

```bash
# Create .env file
cat > .env << 'EOF'
# RAG System Configuration
RAG_LLM_MODEL=llama3.1:8b
RAG_TEMPERATURE=0.1
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_CHROMA_PERSIST_PATH=./data/chroma_db
RAG_PDS_DB_PATH=./data/pds.db
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=64
RAG_TOP_K=5
RAG_USE_RERANKER=false
RAG_DEBUG=true
EOF
```

#### Step 6: Create Data Directories

```bash
mkdir -p data/documents data/chroma_db
```

#### Step 7: Start the Server

```bash
# Development (with hot reload)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

#### Step 8: Verify

```bash
curl http://localhost:8000/api/health
# Expected: {"status":"ok","llm_available":true,"vector_store":{"total_chunks":0}}
```

---

## 4. Local + Cloud LLM (Free API Fallback)

When local hardware is insufficient, use free cloud LLM APIs:

```bash
# .env additions
RAG_LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...  # Get from console.groq.com
```

### Free LLM API Providers

| Provider | Free Models | Rate Limit | Sign Up |
|----------|-----------|------------|---------|
| **Groq** | Llama 3 70B, Mixtral 8x7B, Gemma 2 9B | 30 req/min, 14,400/day | GroqCloud console |
| **Together AI** | Llama 3, Mistral, Qwen | $25 free credit | Together console |
| **OpenRouter** | 200+ models | Pay-per-use, models vary | OpenRouter |
| **HuggingFace** | Many models | 30K chars/month | HF token |
| **GitHub Models** | GPT-4o mini, Llama 3 | Rate limited | GitHub account |

### Hybrid Client

```python
class HybridLLMClient:
    """
    Tries local Ollama first; falls back to free cloud API.
    """

    def __init__(self, ollama_url, groq_api_key=None):
        self.ollama = OllamaClient(ollama_url)
        self.groq = GroqClient(api_key=groq_api_key) if groq_api_key else None

    async def generate(self, prompt: str) -> str:
        # Try local first
        if await self.ollama.is_available():
            return await self.ollama.generate(prompt)
        
        # Fallback to cloud
        if self.groq:
            return await self.groq.generate(prompt)
        
        raise Exception("No LLM available")
```

---

## 5. Cloud VM Deployment (Free Tier)

### Free Cloud Providers

| Provider | Free Tier Specs | Limits | Best For |
|----------|----------------|--------|----------|
| **Oracle Cloud** | 4 OCPU, 24GB RAM, 200GB SSD | Always free | CPU-based RAG |
| **Google Cloud** | 1 vCPU, 1GB RAM (e2-micro) | Per month limit | Light proxy |
| **AWS Free Tier** | 1 vCPU, 1GB RAM (t2.micro) | 750 hrs/month | Not enough for LLM |
| **Azure Free** | 1 vCPU, 1GB RAM (B1s) | 750 hrs/month | Not enough for LLM |

**Oracle Cloud** is the only viable free tier for running a local LLM (24GB RAM).

### Deploy on Oracle Cloud Free Tier

```bash
# 1. Create Oracle Cloud account (requires credit card, never charged)
# 2. Create an AMD VM (4 OCPU, 24GB RAM — always free)
# 3. SSH into the VM
ssh ubuntu@<your-vm-ip>

# 4. Install Ollama
curl -fsSL https://ollama.com/install.sh | sudo sh

# 5. Pull a smaller model (8B won't fit well on CPU, use 3B)
ollama pull llama3.2:3b  

# 6. Install Python + project
sudo apt install python3.11 python3.11-venv git -y
git clone https://github.com/yourusername/ai-rag-system.git
cd ai-rag-system
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# 7. Configure for CPU
cat > .env << 'EOF'
RAG_LLM_MODEL=llama3.2:3b
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_USE_RERANKER=false
RAG_TOP_K=3
EOF

# 8. Run with systemd for auto-restart
sudo tee /etc/systemd/system/rag-api.service << 'SERVICEOF'
[Unit]
Description=AI RAG System API
After=network.target ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-rag-system
Environment=PYTHONPATH=/home/ubuntu/ai-rag-system
ExecStart=/home/ubuntu/ai-rag-system/.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEOF

sudo systemctl daemon-reload
sudo systemctl enable rag-api
sudo systemctl start rag-api
```

---

## 6. Free GPU Options

### RunPod (Serverless GPU)

```bash
# Cheapest GPU: ~$0.20/hr (RTX 3090)
# Best for batch processing, not 24/7

# Pull container
docker pull ollama/ollama

# Run with GPU
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 \
  --name ollama ollama/ollama
```

### Google Colab (Free T4 GPU)

```python
# Run LLM + ChromaDB in Colab for free GPU
# Limited to ~12 hours per session

!pip install ollama chromadb sentence-transformers fastapi uvicorn

# Start Ollama in background
import subprocess
import threading

def run_ollama():
    subprocess.run(["ollama", "serve"])

threading.Thread(target=run_ollama, daemon=True).start()

# Pull model
subprocess.run(["ollama", "pull", "llama3.2:3b"])
```

### HuggingFace Inference Endpoints (Free)

```python
# Use HF Inference API as LLM backend
import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B"
headers = {"Authorization": "Bearer hf_xxxxxxxx"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
```

---

## 7. Cost Breakdown

### Free Tier Components

| Component | Cost | Notes |
|-----------|------|-------|
| **Ollama (local)** | $0 | You already own the hardware |
| **ChromaDB** | $0 | Apache 2.0, unlimited vectors |
| **Embedding (sentence-transformers)** | $0 | Runs on CPU, free models |
| **SQLite** | $0 | Built into Python |
| **FastAPI** | $0 | MIT license |
| **Groq API (fallback)** | $0 | 14,400 requests/day free |
| **Oracle Cloud VM** | $0 | 24GB RAM, 4 OCPU, always free |
| **Google Colab GPU** | $0 | T4 GPU, 12hr sessions |

**Total: $0.00/month for personal use.**

### If You Need More

| Upgrade | Cost | Benefit |
|---------|------|---------|
| **RunPod GPU (spot)** | ~$36/month | 24/7 GPU inference |
| **Together AI API** | ~$0.30/M tokens | Cloud Llama 3 70B speed |
| **Qdrant Cloud** | $0 (1M vectors free) | Managed vector DB |
| **Hetzner VPS (16GB)** | ~€6/month | Dedicated 24/7 server |

---

## 8. Performance Benchmarks

### Expected Latency (End-to-End, Single Query)

| Setup | Embedding | Retrieval | Reranking | LLM Gen | Total |
|-------|-----------|-----------|-----------|---------|-------|
| **CPU (8 core)** | 100ms | 20ms | 500ms | 800ms | ~1.5s |
| **GPU (RTX 3060)** | 30ms | 20ms | 200ms | 200ms | ~0.5s |
| **Groq API** | 100ms | 20ms | — (skip) | 100ms | ~0.3s |
| **Together API** | 100ms | 20ms | — (skip) | 150ms | ~0.3s |
| **Oracle Cloud (CPU)** | 200ms | 30ms | 800ms | 1500ms | ~2.5s |
| **Colab T4 GPU** | 20ms | 20ms | 150ms | 150ms | ~0.4s |

### Throughput

| Setup | Queries/min | Concurrent Users |
|-------|-----------|-----------------|
| CPU, 8-core | 10-15 | 1-2 |
| GPU, RTX 3060 12GB | 30-60 | 3-5 |
| GPU, RTX 4090 24GB | 60-120 | 5-10 |
| Groq API | 30+ | 10+ |

---

## 9. Security Considerations

### Local Deployment (Default)

```
✅ Data never leaves your machine
✅ No network exposure (localhost only)
✅ No API keys required
✅ No third-party access to your documents
```

### Cloud Deployment — Security Checklist

- [ ] **Firewall**: Only expose port 8000. Use UFW:
  ```bash
  sudo ufw allow 22/tcp    # SSH
  sudo ufw allow 8000/tcp  # API (restrict to your IP)
  sudo ufw enable
  ```
- [ ] **Authentication**: Add API key middleware for cloud deployments:
  ```python
  from fastapi import Header, HTTPException

  API_KEY = os.getenv("API_KEY", "")

  async def verify_api_key(x_api_key: str = Header("")):
      if API_KEY and x_api_key != API_KEY:
          raise HTTPException(status_code=403, detail="Invalid API key")
  ```
- [ ] **HTTPS**: Use Caddy or Let's Encrypt for TLS:
  ```bash
  # Caddy automatically provisions HTTPS
  sudo apt install caddy
  sudo tee /etc/caddy/Caddyfile << 'EOF'
  rag.example.com {
      reverse_proxy localhost:8000
  }
  EOF
  ```
- [ ] **Rate limiting**: Prevent abuse on public endpoints:
  ```python
  from slowapi import Limiter
  from slowapi.util import get_remote_address

  limiter = Limiter(key_func=get_remote_address)
  app.state.limiter = limiter

  @app.post("/api/chat/ask")
  @limiter.limit("10/minute")
  async def ask_question(request: ChatRequest):
      ...
  ```
- [ ] **Input sanitization**: Sanitize file uploads:
  ```python
  import magic

  ALLOWED_MIME_TYPES = [
      "text/plain", "text/markdown", "application/pdf",
      "text/csv", "application/json",
  ]

  async def validate_file(file: UploadFile):
      content = await file.read(2048)
      mime = magic.from_buffer(content, mime=True)
      if mime not in ALLOWED_MIME_TYPES:
          raise HTTPException(400, f"Invalid file type: {mime}")
      await file.seek(0)
  ```
- [ ] **Database encryption**: Encrypt SQLite at rest (optional):
  ```python
  # Use sqlcipher instead of sqlite3
  # pip install pysqlcipher3
  from pysqlcipher3 import dbapi2 as sqlite
  conn = sqlite.connect('encrypted.db')
  conn.execute("PRAGMA key='your-passphrase'")
  ```
- [ ] **Logging**: Never log sensitive content:
  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger("rag_api")

  # Sanitize logs
  class SanitizedFilter(logging.Filter):
      def filter(self, record):
          if hasattr(record, 'msg'):
              record.msg = record.msg[:200]  # Truncate
          return True
  ```

---

## 10. Monitoring

### Local Monitoring

```bash
# Check service status
systemctl status ollama
systemctl status rag-api

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor CPU/RAM
htop

# Check disk usage
df -h ./data/

# Tail logs
journalctl -u rag-api -f
tail -f app.log
```

### Health Endpoint (for uptime monitoring)

```bash
# Cron job to check health every 5 minutes
*/5 * * * * curl -s http://localhost:8000/api/health | grep -q "ok" || systemctl restart rag-api
```

---

## 11. Scaling Strategy

```
Phase 1: Local       Phase 2: Hybrid         Phase 3: Distributed
┌──────────┐        ┌─────────────┐         ┌─────────────────┐
│Single user│──────►│Local PDS    │────────►│ChromaDB Cluster │
│1 instance│        │+ Cloud LLM  │         │+ Load Balancer  │
│SQLite    │        │Reranker on  │         │PostgreSQL PDS   │
│ChromaDB  │        │GPU          │         │Auth + Multi-user│
└──────────┘        └─────────────┘         └─────────────────┘
```

### When to Scale

| Signal | Action |
|--------|--------|
| >1 user concurrently | Add API key auth + rate limiting |
| >1000 documents | Add BM25 hybrid search |
| >10,000 chunks | Tune HNSW parameters, add index optimization |
| >100,000 chunks | Migrate from ChromaDB to Qdrant |
| >1M chunks | Add PostgreSQL PDS, shard ChromaDB |
| Response >5s | Switch to GPU LLM or cloud API |

---

## 12. Cost-Saving Tips

1. **Use 4-bit quantized models** — 50% less RAM, 90% of the quality
2. **Batch embeddings** — process chunks in batches of 32-64
3. **Disable reranker** — saves ~500ms per query, 10-15% quality loss
4. **Set `keep_alive: -1`** — avoids reloading model for every query
5. **Use `llama3.2:3b` for CPU** — 3x faster than 8B on CPU
6. **Enable streaming** — user sees first token faster
7. **Set `RAG_USE_RERANKER=false`** on CPU — skip cross-encoder
8. **Cache embeddings** — avoid recomputing for unchanged documents

---

## 13. Disaster Recovery

### Backup Commands

```bash
# Quick backup
tar -czf rag-backup-$(date +%Y%m%d).tar.gz data/

# Auto-backup cron job (daily)
0 3 * * * cd /home/ubuntu/ai-rag-system && \
  tar -czf backups/rag-$(date +\%Y\%m\%d).tar.gz data/ && \
  find backups/ -name "rag-*.tar.gz" -mtime +30 -delete
```

### Restore

```bash
# Stop services
systemctl stop rag-api

# Restore data
tar -xzf rag-backup-20250708.tar.gz

# Restart
systemctl start rag-api
```

---

## 14. Deployment Checklist

- [ ] Ollama installed and running
- [ ] Model pulled (`ollama pull llama3.1:8b`)
- [ ] Python environment created and dependencies installed
- [ ] `.env` file configured
- [ ] Data directories created (`./data/documents`, `./data/chroma_db`)
- [ ] Health check returns `status: "ok"`
- [ ] Document upload works end-to-end
- [ ] Chat query returns answer with sources
- [ ] Streaming endpoint works
- [ ] Firewall configured (if cloud deployment)
- [ ] API key auth added (if public endpoint)
- [ ] HTTPS configured (if public endpoint)
- [ ] Daily backup cron job set up
- [ ] Monitoring (health check, disk space, memory)
- [ ] `keep_alive` set to avoid cold starts
