# 02 — LLM Layer

> **Local, free LLM inference via Ollama — model selection, prompt management, performance tuning**

---

## 1. Why Ollama?

Ollama is the most practical free LLM runner for local RAG systems:

- **Single binary install** — works on macOS, Linux, Windows (WSL2)
- **Auto GPU detection** — uses NVIDIA CUDA, AMD ROCm, or falls back to CPU
- **REST API built-in** — `POST /api/generate` and `POST /api/chat` endpoints
- **Model management** — `ollama pull`, `ollama run`, `ollama rm`
- **Modelfile** — custom prompt templates, system prompts, parameter overrides
- **Open source** — MIT license

### Installation

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com/download
# Or via WSL2:
wsl --install
# Inside WSL:
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version
```

---

## 2. Model Selection Guide

### Recommended Models for Free RAG

| Model | Parameters | RAM | Context | Quality | Speed (CPU) | Speed (GPU) |
|-------|-----------|-----|---------|---------|-------------|-------------|
| **Llama 3.1 8B** | 8B | 8GB | 128K | ★★★★★ | 8-12 tok/s | 40-60 tok/s |
| **Mistral 7B** | 7B | 6GB | 32K | ★★★★ | 10-15 tok/s | 50-70 tok/s |
| **Gemma 2 9B** | 9B | 8GB | 8K | ★★★★ | 7-10 tok/s | 35-50 tok/s |
| **Phi-3 Medium** | 14B | 10GB | 128K | ★★★★ | 4-6 tok/s | 25-35 tok/s |
| **Qwen 2.5 7B** | 7B | 6GB | 128K | ★★★★★ | 10-14 tok/s | 45-65 tok/s |
| **Llama 3.2 3B** | 3B | 4GB | 128K | ★★★ | 20-30 tok/s | 80-100 tok/s |

### Selection Criteria

```
If RAM ≤ 8GB:  → Llama 3.2 3B or Mistral 7B (4-bit quantized)
If RAM ≤ 16GB: → Llama 3.1 8B or Qwen 2.5 7B
If RAM ≥ 32GB: → Qwen 2.5 14B or Mixtral 8x7B
If RAM ≥ 64GB: → Llama 3.1 70B (Q4) or Command R+
```

**Best general-purpose choice:** `llama3.1:8b` (8K context, strong instruction following, Apache 2.0)

**Best for RAG with large docs:** `qwen2.5:7b` (128K context, strong at long-context retrieval)

**Fastest on CPU:** `llama3.2:3b` (3B params, runs comfortably on 4GB RAM)

---

## 3. Pulling and Running Models

```bash
# Pull recommended models
ollama pull llama3.1:8b
ollama pull nomic-embed-text    # For embeddings (optional)
ollama pull llama3.2:3b         # Lightweight fallback

# List pulled models
ollama list

# Test model
ollama run llama3.1:8b "What is Retrieval-Augmented Generation?"
```

### Custom Modelfile (Optimized for RAG)

Create `~/rag-modelfile`:

```
FROM llama3.1:8b

# System prompt optimized for RAG
SYSTEM """You are a precise AI assistant that answers questions based ENTIRELY on the provided context. 

RULES:
1. Answer ONLY using the context provided below. Do not use your internal knowledge.
2. If the context does not contain enough information to answer, say: "I cannot answer this based on the provided documents."
3. Cite the source documents when referencing specific information.
4. If asked about a document that hasn't been uploaded, direct the user to upload it first.
5. Keep responses concise and factual. Do not add opinions or speculation.
6. Format lists, code blocks, and tables when it improves readability.
"""

# Performance tuning for local RAG
PARAMETER temperature 0.1        # Low temp for factual RAG
PARAMETER top_p 0.9
PARAMETER num_ctx 8192          # Context window size
PARAMETER num_predict 1024      # Max response tokens
```

Build and use:
```bash
ollama create rag-assistant -f ~/rag-modelfile
ollama run rag-assistant
```

---

## 4. Programmatic Access via REST API

Ollama exposes a REST API on `http://localhost:11434`.

### Python Client

```python
import httpx
import json
from typing import AsyncGenerator

class OllamaClient:
    """Async client for Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3.1:8b",
                 temperature: float = 0.1):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, prompt: str, system_prompt: str = "",
                       stream: bool = False) -> str | AsyncGenerator[str, None]:
        """Generate a response. Returns string or stream."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_predict": 1024
            }
        }

        if stream:
            return self._stream_generate(payload)
        
        async with self.client.stream("POST",
                f"{self.base_url}/api/generate", json=payload) as resp:
            full_response = []
            async for line in resp.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    full_response.append(data.get("response", ""))
                    if data.get("done", False):
                        break
            return "".join(full_response)

    async def _stream_generate(self, payload: dict) -> AsyncGenerator[str, None]:
        async with self.client.stream("POST",
                f"{self.base_url}/api/generate", json=payload) as resp:
            async for line in resp.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    yield data.get("response", "")

    async def chat(self, messages: list[dict], stream: bool = False):
        """Chat endpoint with message history."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature
            }
        }
        # Similar streaming pattern as generate()
        # Returns {"message": {"role": "assistant", "content": "..."}}

    async def is_available(self) -> bool:
        """Check if Ollama is running and model is loaded."""
        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            models = resp.json().get("models", [])
            return any(m["name"].startswith(self.model) for m in models)
        except Exception:
            return False

    async def close(self):
        await self.client.aclose()
```

### Free API Alternatives (If Local Hardware Is Insufficient)

| Provider | Free Tier | Models | Limits |
|----------|----------|--------|--------|
| **Groq** | Yes | Llama 3 70B, Mixtral 8x7B | 30 req/min, 14,400 req/day |
| **Together AI** | $25 credit | Llama, Mistral, Qwen | One-time credit |
| **HuggingFace Inference API** | Yes | Many models | 30K input chars/month |
| **OpenRouter** | Yes (limited) | Multiple models | Varies by model |

**Architecture note:** The `OllamaClient` class above can be swapped for any API client implementing the same interface. Use a factory pattern:

```python
class LLMFactory:
    @staticmethod
    def get_client(provider: str = "ollama"):
        if provider == "ollama":
            return OllamaClient()
        elif provider == "groq":
            return GroqClient(api_key=settings.GROQ_API_KEY)
        elif provider == "openrouter":
            return OpenRouterClient(api_key=settings.OPENROUTER_KEY)
```

---

## 5. Prompt Template for RAG

The RAG prompt is the single most important factor in response quality.

### Template Structure

```
System: {rag_system_prompt}

--- BEGIN CONTEXT ---
{retrieved_chunks_formatted}
--- END CONTEXT ---

User Question: {query}

Instructions:
- Answer based ONLY on the context above.
- If the context lacks information, say so.
- Cite sources as: [Source: filename.pdf]
```

### Python Implementation

```python
def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build a RAG prompt from query and retrieved chunks.
    
    Args:
        query: User's question
        chunks: List of {"content": str, "metadata": {"source": str, "page": int}}
    
    Returns:
        Formatted prompt string
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page", "")
        page_str = f" (p. {page})" if page else ""
        context_parts.append(
            f"[Document {i}]{page_str}\n{chunk['content']}\n"
        )

    context_block = "\n---\n".join(context_parts)

    prompt = f"""You are a precise AI assistant. Answer the question using ONLY the provided context.

--- BEGIN CONTEXT ---
{context_block}
--- END CONTEXT ---

Question: {query}

Instructions:
1. Answer only from the context above.
2. If unsure, say "I cannot answer this from the provided documents."
3. Cite sources as [Source: filename] when referencing specific information.
4. Be concise and factual.

Answer:"""
    return prompt
```

---

## 6. Context Window Management

This is critical for RAG. The LLM has a fixed context window, and you must fit both retrieved chunks AND the conversation history.

### Strategy: Sliding Window

```
┌──────────────────────────────────┐
│         Context Window           │
│         (8192 tokens)            │
├──────────────────────────────────┤
│                                  │
│  System Prompt      ~500 tokens  │
│                                  │
│  Conversation Hist  ~1000 tokens │
│  (last N turns)                  │
│                                  │
│  Retrieved Chunks   ~5000 tokens │
│  (top-K chunks)                  │
│                                  │
│  User Query          ~200 tokens │
│                                  │
│  Generation Budget  ~1500 tokens │
│  (response)                      │
│                                  │
└──────────────────────────────────┘
```

### Budget Calculation

```python
import tiktoken  # or any tokenizer

def calculate_chunk_budget(
    model_context: int = 8192,
    system_prompt: str = "",
    conversation_history: list[dict] = None,
    query: str = "",
    reserve_response: int = 1500
) -> int:
    """Calculate how many tokens are available for context chunks."""
    enc = tiktoken.get_encoding("cl100k_base")

    used = 0
    used += len(enc.encode(system_prompt))
    
    if conversation_history:
        for msg in conversation_history[-6:]:  # Last 6 turns
            used += len(enc.encode(msg.get("content", "")))
    
    used += len(enc.encode(query))
    used += reserve_response
    used += 200  # Overhead for formatting, separators, etc.

    return max(0, model_context - used)
```

---

## 7. Performance Optimization

### Quantization Levels

| Quantization | Model Size | Quality Loss | Speed Gain | RAM |
|-------------|-----------|-------------|------------|-----|
| Q4_K_M | 4.9GB | Minimal | ~1.5x | 8GB |
| Q5_K_M | 5.7GB | Very minor | ~1.3x | 10GB |
| Q8_0 | 8.5GB | Negligible | ~1.1x | 12GB |
| F16 | 16GB | None | 1x | 20GB+ |

```bash
# Pull a quantized model
ollama pull llama3.1:8b-q4_K_M   # ~4.9GB - sweet spot
```

### Concurrent Requests

Ollama queues requests by default. For multiple users, consider:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OllamaPool:
    """Simple request pool for Ollama."""
    
    def __init__(self, max_concurrent: int = 1):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_limit(self, prompt: str) -> str:
        async with self.semaphore:
            return await self.client.generate(prompt)
```

### Keep Model Loaded (Prevent Cold Starts)

```bash
# Preload model into GPU memory
ollama run llama3.1:8b &

# Or via API
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "llama3.1:8b", "prompt": "warmup", "keep_alive": -1}'
```

Set `keep_alive: -1` in API calls to keep the model loaded between requests.

---

## 8. Error Handling

```python
class LLMException(Exception):
    """Base exception for LLM failures."""

class ModelNotAvailable(LLMException):
    """Model is not pulled or Ollama is not running."""

class ContextOverflow(LLMException):
    """Prompt exceeds model's context window."""

async def safe_generate(client: OllamaClient, prompt: str) -> str:
    """Generate with fallback and error handling."""
    try:
        if not await client.is_available():
            raise ModelNotAvailable("Ollama not running or model missing")
        return await client.generate(prompt)
    except httpx.TimeoutException:
        return "Response timed out. Try a simpler question."
    except ModelNotAvailable:
        return "Local LLM is not available. Please start Ollama."
    except Exception as e:
        return f"Generation failed: {str(e)}"
```

---

## 9. Monitoring

```bash
# Ollama logs
ollama serve &  # Run in foreground to see logs
journalctl -u ollama  # On Linux with systemd

# Check GPU usage
nvidia-smi -l 1  # GPU memory, utilization

# Monitor via API
curl http://localhost:11434/api/ps  # Running models
```

---

## 10. Checklist

- [ ] Ollama installed and running
- [ ] At least one model pulled (`ollama pull llama3.1:8b`)
- [ ] Modelfile customized for RAG (low temperature, source citation)
- [ ] Python client tested with `is_available()`
- [ ] Context window budget calculated for your model
- [ ] Streaming responses working from API
- [ ] Fallback provider configured (if using free cloud API)
- [ ] `keep_alive` set to avoid cold starts
