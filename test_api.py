"""API integration tests — runs against live server at localhost:8000."""
import json, sys, os
import httpx

BASE = "http://localhost:8000/api"
pass_count = 0
fail_count = 0

def test(name, fn):
    global pass_count, fail_count
    try:
        fn()
        print(f"  [PASS] {name}")
        pass_count += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        fail_count += 1

client = httpx.Client(base_url=BASE, timeout=30)

# ── Cleanup any leftover documents from previous runs ──
def cleanup_old_docs():
    r = client.get("/documents/")
    if r.status_code == 200:
        for doc in r.json():
            client.delete(f"/documents/{doc['id']}")
cleanup_old_docs()

# ── Health ──

def health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "running"

test("GET /api/health returns running", health)

def admin_health():
    r = client.get("/admin/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "components" in data
    assert "llm" in data["components"]

test("GET /api/admin/health returns status", admin_health)

def admin_stats():
    r = client.get("/admin/stats")
    assert r.status_code == 200
    data = r.json()
    assert "documents" in data
    assert "vectors" in data

test("GET /api/admin/stats returns stats", admin_stats)

# ── Document Upload ──

def upload_text_file():
    content = b"The capital of France is Paris. The Eiffel Tower is located in Paris."
    r = client.post("/documents/upload", files={
        "file": ("test.txt", content, "text/plain"),
    }, data={"category": "geography", "tags": "france,paris"})
    assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
    data = r.json()
    assert data["filename"] == "test.txt"
    assert data["chunks"] >= 1
    assert data["category"] == "geography"
    # Store for later tests
    test.uploaded_id = data["document_id"]
    print(f"      Uploaded doc ID: {data['document_id']}, chunks: {data['chunks']}")

def upload_markdown():
    content = b"# Meeting Notes\n\nDiscussed Q3 budget.\n## Action Items\n- Finalise report\n- Send invoice"
    r = client.post("/documents/upload", files={
        "file": ("notes.md", content, "text/markdown"),
    })
    assert r.status_code == 200
    test.uploaded_md_id = r.json()["document_id"]

test("POST /documents/upload — text file", upload_text_file)
test("POST /documents/upload — markdown", upload_markdown)

# ── Document List ──

def list_documents():
    r = client.get("/documents/")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 2
    ids = [d["id"] for d in data]
    assert test.uploaded_id in ids

test("GET /documents/ lists all documents", list_documents)

def get_document():
    r = client.get(f"/documents/{test.uploaded_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "test.txt"
    assert data["file_type"] == "txt"

test("GET /documents/{id} returns document", get_document)

def doc_stats():
    r = client.get("/documents/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_documents"] >= 2
    assert data["vector_chunks"] >= 2

test("GET /documents/stats returns counts", doc_stats)

# ── Chat / RAG ──

def chat_session_create():
    r = client.post("/chat/sessions", params={"title": "Test Session"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    test.session_id = data["session_id"]
    print(f"      Session ID: {data['session_id']}")

def chat_list_sessions():
    r = client.get("/chat/sessions")
    assert r.status_code == 200
    sessions = r.json()
    assert len(sessions) >= 1

def chat_ask():
    r = client.post("/chat/ask", json={
        "query": "What is the capital of France?",
        "session_id": test.session_id,
    })
    assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
    data = r.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "Paris" in data["answer"] or "capital" in data["answer"], \
        f"Answer should reference Paris: {data['answer']}"
    assert len(data["sources"]) > 0
    assert "latency_ms" in data
    print(f"      Answer: \"{data['answer'][:100]}...\"")
    print(f"      Sources: {len(data['sources'])} chunks")
    print(f"      Latency: {data['latency_ms']}ms")
    test.last_answer = data["answer"]
    test.last_sources = data["sources"]

def chat_ask_with_filter():
    r = client.post("/chat/ask", json={
        "query": "What are the action items?",
        "document_filter": "notes.md",
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["answer"]) > 0
    print(f"      Filtered answer: \"{data['answer'][:100]}...\"")

def chat_messages():
    r = client.get(f"/chat/sessions/{test.session_id}/messages")
    assert r.status_code == 200
    msgs = r.json()
    assert len(msgs) >= 2  # user + assistant
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"

test("POST /chat/sessions creates session", chat_session_create)
test("GET /chat/sessions lists sessions", chat_list_sessions)
test("POST /chat/ask returns answer with RAG", chat_ask)
test("POST /chat/ask with document filter", chat_ask_with_filter)
test("GET /chat/sessions/{id}/messages returns history", chat_messages)

# ── Streaming ──

def chat_stream():
    r = client.post("/chat/ask/stream", json={
        "query": "Say hello world in 3 words.",
        "session_id": test.session_id,
        "stream": True,
    })
    assert r.status_code == 200
    events = []
    for line in r.iter_lines():
        if line.startswith("event:"):
            events.append(line)
    assert len(events) >= 1
    print(f"      Received {len(events)} SSE events")

test("POST /chat/ask/stream returns SSE events", chat_stream)

# ── Delete ──

def delete_document():
    r = client.delete(f"/documents/{test.uploaded_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "deleted"
    # Verify it's gone
    r2 = client.get(f"/documents/{test.uploaded_id}")
    assert r2.status_code == 404

def delete_nonexistent():
    r = client.delete("/documents/nonexistent-id")
    assert r.status_code == 404

test("DELETE /documents/{id} removes document", delete_document)
test("DELETE /documents/{id} returns 404 for missing", delete_nonexistent)

# ── Error Handling ──

def upload_invalid_type():
    r = client.post("/documents/upload", files={
        "file": ("bad.exe", b"data", "application/octet-stream"),
    })
    assert r.status_code == 400

def chat_empty_query():
    r = client.post("/chat/ask", json={"query": ""})
    # Should not crash — might return 200 or 422 depending on validation
    assert r.status_code in (200, 422)

def session_not_found():
    r = client.post("/chat/ask", json={
        "query": "test",
        "session_id": "doesnotexist",
    })
    assert r.status_code == 404

test("Upload invalid file type returns 400", upload_invalid_type)
test("Empty query does not crash server", chat_empty_query)
test("Non-existent session returns 404", session_not_found)

# ── Summary ──

print()
print(f"RESULTS: {pass_count} passed, {fail_count} failed, {pass_count + fail_count} total")
sys.exit(0 if fail_count == 0 else 1)
