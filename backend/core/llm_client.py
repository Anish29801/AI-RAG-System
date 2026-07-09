"""LLM client abstraction.

Wraps the Ollama REST API with async support, streaming, and
keep-alive management. Designed as a drop-in — swap the backend
by implementing the same interface for any provider.
"""

import json
from typing import AsyncGenerator, Optional

import httpx

from backend.config import settings


class LLMClient:
    """Async client for Ollama LLM API with adjustable generation parameters."""

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 40,
        timeout: float = 120.0,
    ):
        self.base_url = base_url or settings.ollama_url
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.top_p = top_p if top_p is not None else settings.top_p
        self.top_k = top_k if top_k is not None else settings.llm_top_k
        self._client = httpx.AsyncClient(timeout=timeout)

    # ── Generation ──

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Generate a response. Returns string or async token generator."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_predict": 1024,
                "keep_alive": -1,
            },
        }

        if stream:
            return self._stream(payload)

        return await self._generate_full(payload)

    async def _generate_full(self, payload: dict) -> str:
        parts = []
        async with self._client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    parts.append(data.get("response", ""))
        return "".join(parts)

    async def _stream(self, payload: dict) -> AsyncGenerator[str, None]:
        async with self._client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    yield data.get("response", "")

    # ── Chat (conversation mode) ──

    async def chat(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Chat endpoint with message history."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "keep_alive": -1,
            },
        }

        if stream:
            return self._chat_stream(payload)

        return await self._chat_full(payload)

    async def _chat_full(self, payload: dict) -> str:
        async with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            parts = []
            async for line in resp.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    parts.append(data["message"]["content"])
            return "".join(parts)

    async def _chat_stream(self, payload: dict) -> AsyncGenerator[str, None]:
        async with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    yield data["message"]["content"]

    # ── Settings ──

    def get_settings(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def update_settings(self, **kwargs):
        valid = {"model", "temperature", "top_p", "top_k"}
        for k, v in kwargs.items():
            if k in valid and v is not None:
                setattr(self, k, v)

    # ── Models ──

    async def list_models(self) -> list[dict]:
        """Fetch available models from Ollama."""
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            models = resp.json().get("models", [])
            return [
                {
                    "name": m["name"],
                    "size_bytes": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in models
            ]
        except Exception:
            return []

    # ── Health ──

    async def is_available(self) -> bool:
        """Check Ollama is running and the model is available."""
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            models = resp.json().get("models", [])
            return any(m["name"].startswith(self.model) for m in models)
        except Exception:
            return False

    # ── Cleanup ──

    async def close(self):
        await self._client.aclose()
