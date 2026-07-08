"""Batch 4 — LLM Layer Tests.

Sub-batch 4A: Unit tests with mocked httpx (no Ollama required)
Sub-batch 4B: Live integration against real Ollama (requires Ollama)

Run unit:   pytest tests/test_llm.py -v -m "not integration"
Run live:   pytest tests/test_llm.py -v -m "integration"
Run all:    pytest tests/test_llm.py -v
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx


pytestmark = pytest.mark.asyncio


# ═══════════════════════════════════════════════════════════════
# Sub-batch 4A — Mocked LLMClient
# ═══════════════════════════════════════════════════════════════


class MockStreamResponse:
    """Simulates httpx streaming response for async context manager."""

    def __init__(self, lines: list[dict], status: int = 200):
        self._lines = [json.dumps(line) for line in lines]
        self._status = status
        self._index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        httpx_status = self._status
        msg = {400: "Bad Request", 500: "Internal Server Error"}.get(httpx_status, "Error")
        if httpx_status >= 400:
            raise httpx.HTTPStatusError(msg, request=MagicMock(), response=self)

    @property
    def status_code(self):
        return self._status

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        val = self._lines[self._index]
        self._index += 1
        return val


class MockGetResponse:
    """Simulates httpx GET response (json() is sync in httpx)."""

    def __init__(self, json_data: dict, status: int = 200):
        self._json_data = json_data
        self._status = status

    def json(self):
        return self._json_data

    @property
    def status_code(self):
        return self._status


@pytest.fixture
def client():
    """Patch httpx.AsyncClient so LLMClient uses a controlled mock.

    stream(): MagicMock → returns MockStreamResponse (async with works directly)
    get():    AsyncMock  → awaitable, returns MockGetResponse
    aclose(): AsyncMock  → awaitable for LLMClient.close()
    """
    with patch("backend.core.llm_client.httpx.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.stream = MagicMock()
        mock_client.get = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_cls.return_value = mock_client
        yield mock_client


class TestLLMClientMocked:
    """LLMClient unit tests with mocked httpx."""

    async def test_generate_sends_correct_payload(self, client):
        response = MockStreamResponse([{"response": "Hello!"}])
        client.stream.return_value = response

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434", model="test-model")
        result = await llm.generate("Say hello")

        call_args, call_kwargs = client.stream.call_args
        assert call_args[1] == "http://test:11434/api/generate"
        payload = call_kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["prompt"] == "Say hello"
        assert payload["stream"] is False
        assert result == "Hello!"
        await llm.close()

    async def test_generate_stream_returns_async_gen(self, client):
        response = MockStreamResponse([
            {"response": "Hello "},
            {"response": "World!"},
        ])
        client.stream.return_value = response

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434")
        gen = await llm.generate("test", stream=True)
        tokens = []
        async for token in gen:
            tokens.append(token)
        assert "".join(tokens) == "Hello World!"
        await llm.close()

    async def test_generate_includes_system_prompt(self, client):
        response = MockStreamResponse([{"response": "OK"}])
        client.stream.return_value = response

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434")
        await llm.generate("hi", system_prompt="You are helpful.")
        payload = client.stream.call_args[1]["json"]
        assert payload["system"] == "You are helpful."
        await llm.close()

    async def test_generate_sets_options(self, client):
        response = MockStreamResponse([{"response": "OK"}])
        client.stream.return_value = response

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434", temperature=0.5)
        await llm.generate("hi")
        opts = client.stream.call_args[1]["json"]["options"]
        assert opts["temperature"] == 0.5
        assert opts["top_p"] == 0.9
        assert opts["num_predict"] == 1024
        assert opts["keep_alive"] == -1
        await llm.close()

    async def test_chat_sends_messages_array(self, client):
        response = MockStreamResponse([{"message": {"content": "Hi there!"}}])
        client.stream.return_value = response

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434")
        result = await llm.chat([{"role": "user", "content": "Hello"}])
        payload = client.stream.call_args[1]["json"]
        assert "messages" in payload
        assert payload["messages"][0]["content"] == "Hello"
        assert result == "Hi there!"
        await llm.close()

    async def test_chat_stream_returns_tokens(self, client):
        response = MockStreamResponse([
            {"message": {"content": "Hello "}},
            {"message": {"content": "World"}},
        ])
        client.stream.return_value = response

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434")
        gen = await llm.chat([{"role": "user", "content": "Hi"}], stream=True)
        tokens = []
        async for token in gen:
            tokens.append(token)
        assert "".join(tokens) == "Hello World"
        await llm.close()

    async def test_is_available_returns_true(self, client):
        client.get.return_value = MockGetResponse({
            "models": [{"name": "test-model:latest"}],
        })

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434", model="test-model")
        avail = await llm.is_available()
        assert avail is True
        assert client.get.call_args[0][0] == "http://test:11434/api/tags"
        await llm.close()

    async def test_is_available_returns_false_on_http_error(self, client):
        client.get.side_effect = Exception("Connection refused")

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434")
        avail = await llm.is_available()
        assert avail is False
        await llm.close()

    async def test_is_available_returns_false_model_missing(self, client):
        client.get.return_value = MockGetResponse({"models": [{"name": "other-model"}]})

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434", model="missing-model")
        avail = await llm.is_available()
        assert avail is False
        await llm.close()

    async def test_generate_handles_http_error(self, client):
        response = MockStreamResponse([], status=500)
        client.stream.return_value = response

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434")
        with pytest.raises(Exception):
            await llm.generate("test")
        await llm.close()

    async def test_generate_handles_malformed_json(self, client):
        response = MockStreamResponse([])
        async def bad_lines():
            yield "not-json"
        response.aiter_lines = bad_lines
        client.stream.return_value = response

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434")
        with pytest.raises(json.JSONDecodeError):
            await llm.generate("test")
        await llm.close()

    async def test_close_disposes_client(self, client):
        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434")
        await llm.close()
        client.aclose.assert_awaited_once()

    async def test_connection_reuse(self, client):
        response = MockStreamResponse([{"response": "OK"}])
        client.stream.return_value = response

        from backend.core.llm_client import LLMClient
        llm = LLMClient(base_url="http://test:11434")
        await llm.generate("q1")
        await llm.generate("q2")
        assert client.stream.call_count == 2
        await llm.close()


# ═══════════════════════════════════════════════════════════════
# Sub-batch 4B — Live Integration (requires Ollama)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestLLMClientLive:
    """Live integration tests against real Ollama instance."""

    async def test_is_available_live(self):
        from backend.core.llm_client import LLMClient
        llm = LLMClient()
        avail = await llm.is_available()
        assert avail, "Ollama must be running for integration tests"
        await llm.close()

    async def test_generate_live(self):
        from backend.core.llm_client import LLMClient
        llm = LLMClient()
        resp = await llm.generate("Say 'OK' in one word.")
        assert resp.strip(), "Empty LLM response"
        await llm.close()

    async def test_generate_stream_live(self):
        from backend.core.llm_client import LLMClient
        llm = LLMClient()
        gen = await llm.generate("Say 'OK'", stream=True)
        tokens = []
        async for token in gen:
            tokens.append(token)
        assert len(tokens) > 0
        await llm.close()

    async def test_chat_live(self):
        from backend.core.llm_client import LLMClient
        llm = LLMClient()
        resp = await llm.chat([{"role": "user", "content": "Say 'OK'"}])
        assert resp.strip(), "Empty chat response"
        await llm.close()
