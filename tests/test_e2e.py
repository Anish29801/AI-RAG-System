"""Batch 5 — End-to-End System Tests.

Tests the full system golden path — upload → index → query → verify.
Requires live server (localhost:8000) AND Ollama running.

Start:  uvicorn backend.main:app
Run:    pytest tests/test_e2e.py -v -m e2e
"""

import os
import sys
import json
from pathlib import Path

import pytest
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://localhost:8000/api"
TIMEOUT = 120


# ── Fixtures ──


@pytest.fixture(scope="module")
def client():
    c = httpx.Client(base_url=BASE, timeout=TIMEOUT)
    yield c
    c.close()


def _upload_test_doc(client, filename: str, content: str, category="test") -> dict:
    """Helper: upload a text document and return the JSON response."""
    r = client.post(
        "/documents/upload",
        files={"file": (filename, content.encode(), "text/plain")},
        data={"category": category},
    )
    assert r.status_code == 200, f"Upload failed: {r.text}"
    return r.json()


def _cleanup_all(client):
    """Remove all test documents."""
    r = client.get("/documents/")
    if r.status_code != 200:
        return
    for doc in r.json():
        client.delete(f"/documents/{doc['id']}")


@pytest.mark.e2e
class TestGoldenPath:
    """Full golden path: upload → list → query → verify."""

    def test_health_check(self, client):
        r = client.get("/admin/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("healthy", "degraded")
        assert "components" in data
        assert data["components"]["llm"]["available"] is True

    def test_upload_and_list(self, client):
        _cleanup_all(client)
        resp = _upload_test_doc(client, "e2e_test.txt", "Paris is the capital of France.")
        doc_id = resp["document_id"]
        assert resp["chunks"] >= 1
        assert resp["filename"] == "e2e_test.txt"

        r = client.get("/documents/")
        assert r.status_code == 200
        docs = r.json()
        ids = [d["id"] for d in docs]
        assert doc_id in ids

    def test_ask_with_answer(self, client):
        _upload_test_doc(client, "e2e_qa.txt", "The Eiffel Tower is in Paris. It was built in 1889.")
        r = client.post(
            "/chat/ask",
            json={"query": "Where is the Eiffel Tower?", "stream": False},
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert len(data["sources"]) > 0
        assert "session_id" in data
        assert data["latency_ms"] > 0

    def test_answer_has_relevant_sources(self, client):
        _upload_test_doc(client, "e2e_geo.txt", "Rome is the capital of Italy. The Colosseum is in Rome.")
        r = client.post(
            "/chat/ask",
            json={"query": "What is the capital of Italy?", "stream": False},
        )
        data = r.json()
        sources = data["sources"]
        assert len(sources) >= 1
        assert any("Italy" in s.get("content_preview", "") for s in sources)

    def test_streaming_response(self, client):
        _upload_test_doc(client, "e2e_stream.txt", "Python is a programming language created by Guido van Rossum.")
        r = client.post(
            "/chat/ask/stream",
            json={"query": "Who created Python?", "stream": True},
        )
        assert r.status_code == 200
        events = 0
        tokens = 0
        for line in r.iter_lines():
            if line.startswith("event:"):
                pass
            elif line.startswith("data:"):
                events += 1
                payload = json.loads(line[5:])
                if "token" in payload:
                    tokens += 1
        assert events > 0, "No SSE events received"
        assert tokens > 0, "No stream tokens received"


@pytest.mark.e2e
class TestSessionWorkflow:
    """Session creation, persistence, and message history."""

    def test_create_session(self, client):
        r = client.post("/chat/sessions", params={"title": "E2E Test Session"})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["title"] == "E2E Test Session"

    def test_list_sessions(self, client):
        r = client.get("/chat/sessions")
        assert r.status_code == 200
        sessions = r.json()
        assert isinstance(sessions, list)

    def test_session_persists_messages(self, client):
        _upload_test_doc(client, "e2e_sesh.txt", "Gravity accelerates objects at 9.8 m/s².")

        # Create session
        r = client.post("/chat/sessions", params={"title": "Physics Chat"})
        session_id = r.json()["session_id"]

        # Ask with session
        r = client.post(
            "/chat/ask",
            json={"query": "What is gravity?", "session_id": session_id, "stream": False},
        )
        assert r.status_code == 200

        # Check messages persisted
        r = client.get(f"/chat/sessions/{session_id}/messages")
        assert r.status_code == 200
        msgs = r.json()
        assert len(msgs) >= 2
        assert msgs[0]["role"] == "user"
        assert msgs[-1]["role"] == "assistant"
        assert len(msgs[-1].get("sources", [])) > 0


@pytest.mark.e2e
class TestDocumentCRUD:
    """Create, read, update (via reindex), delete."""

    def test_get_single_document(self, client):
        _cleanup_all(client)
        resp = _upload_test_doc(client, "e2e_crud.txt", "Content for CRUD test.")
        doc_id = resp["document_id"]

        r = client.get(f"/documents/{doc_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == doc_id
        assert "content" not in data  # metadata-only endpoint

    def test_delete_document(self, client):
        resp = _upload_test_doc(client, "e2e_delete.txt", "This will be deleted.")
        doc_id = resp["document_id"]

        r = client.delete(f"/documents/{doc_id}")
        assert r.status_code == 200

        r = client.get(f"/documents/{doc_id}")
        assert r.status_code == 404

    def test_document_stats(self, client):
        r = client.get("/documents/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_documents" in data
        assert "vector_chunks" in data

    def test_reindex(self, client):
        _upload_test_doc(client, "e2e_reindex.txt", "Reindex test content for verification.")
        r = client.post("/admin/reindex")
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert len(data["results"]) > 0
        for result in data["results"]:
            assert result["status"] in ("reindexed", "file_missing", "failed")


@pytest.mark.e2e
class TestErrorHandling:
    """API error cases and edge conditions."""

    def test_nonexistent_document(self, client):
        r = client.get("/documents/nonexistent-id-12345")
        assert r.status_code == 404

    def test_delete_nonexistent(self, client):
        r = client.delete("/documents/nonexistent-id-12345")
        assert r.status_code == 404

    def test_empty_query(self, client):
        r = client.post(
            "/chat/ask",
            json={"query": "", "stream": False},
        )
        # Should either accept (returns generic answer) or reject
        assert r.status_code in (200, 422)

    def test_nonexistent_session(self, client):
        r = client.post(
            "/chat/ask",
            json={"query": "Hello", "session_id": "fake-session-id", "stream": False},
        )
        assert r.status_code == 404

    def test_invalid_file_type(self, client):
        r = client.post(
            "/documents/upload",
            files={"file": ("test.exe", b"fake PE binary", "application/x-msdownload")},
            data={"category": "test"},
        )
        assert r.status_code == 400

    def test_health_degraded_info(self, client):
        r = client.get("/admin/health")
        data = r.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_root_health(self, client):
        r = client.get("http://localhost:8000/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "running"


@pytest.mark.e2e
class TestDegradedRecovery:
    """System behaviour under edge conditions."""

    def test_upload_then_query_twice(self, client):
        """Same query after upload should work and return consistent shape."""
        _upload_test_doc(client, "e2e_repeat.txt", "Berlin is the capital of Germany.")
        r1 = client.post("/chat/ask", json={"query": "What is Berlin?", "stream": False})
        assert r1.status_code == 200
        r2 = client.post("/chat/ask", json={"query": "What is Berlin?", "stream": False})
        assert r2.status_code == 200
        assert "answer" in r2.json()

    def test_upload_large_text(self, client):
        """Chunker handles moderately large text via the API."""
        text = "Python is great. " * 5000
        resp = _upload_test_doc(client, "e2e_large.txt", text)
        assert resp["characters"] > 0
        assert resp["chunks"] > 1

    def test_stats_after_operations(self, client):
        """Stats endpoint returns updated counts after document operations."""
        r = client.get("/documents/stats")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data.get("total_documents", 0), int)
        assert isinstance(data.get("vector_chunks", 0), int)
