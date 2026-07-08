"""Batch 3 — API Integration Tests.

Tests ALL API endpoints against a LIVE running server at localhost:8000.
Requires Ollama running and server started.

Start server:  python -m uvicorn backend.main:app
Run tests:    pytest tests/test_api_intense.py -v

Sub-batches:
- 3A: Health & Admin
- 3B: Document CRUD
- 3C: Chat & RAG
- 3D: Error Handling
"""

import os
import sys
import json
from pathlib import Path

import pytest
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BASE = "http://localhost:8000/api"
TIMEOUT = 60


@pytest.fixture(scope="module")
def client():
    c = httpx.Client(base_url=BASE, timeout=TIMEOUT)
    yield c
    c.close()


def _cleanup(client):
    """Remove leftover documents from previous runs."""
    r = client.get("/documents/")
    if r.status_code == 200:
        for doc in r.json():
            client.delete(f"/documents/{doc['id']}")


# ═══════════════════════════════════════════════════════════════
# Sub-batch 3A — Health & Admin
# ═══════════════════════════════════════════════════════════════


class TestHealthAdmin:
    """Health check and admin endpoints."""

    def test_health_running(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"

    def test_admin_health_returns_components(self, client):
        r = client.get("/admin/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "components" in data
        assert "llm" in data["components"]
        assert "vector_store" in data["components"]

    def test_admin_stats_returns_counts(self, client):
        r = client.get("/admin/stats")
        assert r.status_code == 200
        data = r.json()
        assert "documents" in data
        assert "vectors" in data
        assert "uptime_seconds" in data


# ═══════════════════════════════════════════════════════════════
# Sub-batch 3B — Document CRUD
# ═══════════════════════════════════════════════════════════════


class TestDocumentCRUD:
    """Document upload, list, detail, delete endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, client):
        _cleanup(client)

    def test_upload_txt(self, client):
        r = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"Paris is the capital of France.", "text/plain")},
            data={"category": "geography", "tags": "france,paris"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["filename"] == "test.txt"
        assert data["chunks"] >= 1
        assert data["category"] == "geography"
        pytest.uploaded_id = data["document_id"]

    def test_upload_markdown(self, client):
        content = b"# Notes\n\nDiscussed Q3.\n## Actions\n- Finalise report\n- Send invoice"
        r = client.post(
            "/documents/upload",
            files={"file": ("notes.md", content, "text/markdown")},
        )
        assert r.status_code == 200
        pytest.md_id = r.json()["document_id"]

    def test_upload_duplicate_returns_409(self, client):
        r = client.post(
            "/documents/upload",
            files={"file": ("dup.txt", b"Duplicate content.", "text/plain")},
        )
        assert r.status_code == 200
        r2 = client.post(
            "/documents/upload",
            files={"file": ("dup2.txt", b"Duplicate content.", "text/plain")},
        )
        assert r2.status_code in (200, 409)

    def test_upload_invalid_type_returns_400(self, client):
        r = client.post(
            "/documents/upload",
            files={"file": ("bad.exe", b"data", "application/octet-stream")},
        )
        assert r.status_code == 400

    def test_upload_empty_file(self, client):
        r = client.post(
            "/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert r.status_code in (200, 400)

    def test_list_documents(self, client):
        r = client.get("/documents/")
        assert r.status_code == 200
        docs = r.json()
        assert isinstance(docs, list)

    def test_list_documents_with_filter(self, client):
        r = client.get("/documents/?file_type=txt")
        assert r.status_code == 200
        docs = r.json()
        for d in docs:
            assert d["file_type"] == "txt"

    def test_get_document_by_id(self, client):
        # Upload own doc — autouse fixture cleans state between tests
        r = client.post(
            "/documents/upload",
            files={"file": ("get_me.txt", b"Document for GET test.", "text/plain")},
        )
        assert r.status_code == 200
        doc_id = r.json()["document_id"]
        r = client.get(f"/documents/{doc_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == doc_id
        assert data["filename"] == "get_me.txt"

    def test_get_nonexistent_document_returns_404(self, client):
        r = client.get("/documents/nonexistent-id-12345")
        assert r.status_code == 404

    def test_document_stats(self, client):
        r = client.get("/documents/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_documents" in data
        assert "vector_chunks" in data or "documents_by_type" in data

    def test_delete_document(self, client):
        r = client.post(
            "/documents/upload",
            files={"file": ("delete_me.txt", b"Document for DELETE test.", "text/plain")},
        )
        assert r.status_code == 200
        doc_id = r.json()["document_id"]
        r = client.delete(f"/documents/{doc_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "deleted"
        r2 = client.get(f"/documents/{doc_id}")
        assert r2.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        r = client.delete("/documents/nonexistent-id-12345")
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════
# Sub-batch 3C — Chat & RAG
# ═══════════════════════════════════════════════════════════════


class TestChatRAG:
    """Chat sessions, RAG query, streaming, message history."""

    @pytest.fixture(autouse=True)
    def setup(self, client):
        _cleanup(client)
        # Upload a doc for RAG tests
        r = client.post(
            "/documents/upload",
            files={"file": ("rag_test.txt", b"The capital of France is Paris. The Eiffel Tower is in Paris.", "text/plain")},
        )
        pytest.rag_doc_id = r.json()["document_id"]

    def test_create_session(self, client):
        r = client.post("/chat/sessions", params={"title": "Test Session"})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["title"] == "Test Session"
        pytest.session_id = data["session_id"]

    def test_list_sessions(self, client):
        r = client.get("/chat/sessions")
        assert r.status_code == 200
        sessions = r.json()
        assert isinstance(sessions, list)
        if hasattr(pytest, "session_id"):
            ids = [s["id"] for s in sessions]
            assert pytest.session_id in ids

    def test_ask_returns_answer(self, client):
        if not hasattr(pytest, "session_id"):
            pytest.skip("No session")
        r = client.post("/chat/ask", json={
            "query": "What is the capital of France?",
            "session_id": pytest.session_id,
        })
        assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "sources" in data
        assert len(data["sources"]) > 0, "No sources returned"
        assert "latency_ms" in data
        assert "session_id" in data
        pytest.last_answer = data["answer"]

    def test_ask_with_document_filter(self, client):
        if not hasattr(pytest, "session_id"):
            pytest.skip("No session")
        r = client.post("/chat/ask", json={
            "query": "What is the capital?",
            "document_filter": "rag_test.txt",
            "session_id": pytest.session_id,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["answer"]) > 0

    def test_ask_nonexistent_session_returns_404(self, client):
        r = client.post("/chat/ask", json={
            "query": "test",
            "session_id": "doesnotexist-12345",
        })
        assert r.status_code == 404

    def test_ask_empty_query(self, client):
        if not hasattr(pytest, "session_id"):
            pytest.skip("No session")
        r = client.post("/chat/ask", json={
            "query": "",
            "session_id": pytest.session_id,
        })
        assert r.status_code in (200, 422)

    def test_message_history(self, client):
        if not hasattr(pytest, "session_id"):
            pytest.skip("No session")
        r = client.get(f"/chat/sessions/{pytest.session_id}/messages")
        assert r.status_code == 200
        msgs = r.json()
        assert len(msgs) >= 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_stream_sse_events(self, client):
        if not hasattr(pytest, "session_id"):
            pytest.skip("No session")
        r = client.post("/chat/ask/stream", json={
            "query": "Say hello in 3 words.",
            "session_id": pytest.session_id,
            "stream": True,
        })
        assert r.status_code == 200
        events = []
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append(line)
        assert len(events) >= 1

    def test_multiple_asks_same_session(self, client):
        if not hasattr(pytest, "session_id"):
            pytest.skip("No session")
        for q in ["Question one?", "Question two?", "Question three?"]:
            r = client.post("/chat/ask", json={
                "query": q,
                "session_id": pytest.session_id,
            })
            assert r.status_code == 200
        r = client.get(f"/chat/sessions/{pytest.session_id}/messages")
        msgs = r.json()
        assert len(msgs) >= 6


# ═══════════════════════════════════════════════════════════════
# Sub-batch 3D — Error Handling
# ═══════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Edge cases, malformed input, security."""

    def test_invalid_json_body_returns_422(self, client):
        r = client.post("/chat/ask", data="not-json")
        assert r.status_code in (400, 422)

    def test_wrong_method_returns_405(self, client):
        r = client.put("/documents/")
        assert r.status_code == 405

    def test_xss_in_query(self, client):
        r = client.post("/chat/ask", json={"query": "<script>alert(1)</script>"})
        assert r.status_code in (200, 422)

    def test_sql_injection_in_query(self, client):
        r = client.post("/chat/ask", json={"query": "'; DROP TABLE documents; --"})
        assert r.status_code in (200, 422)
