# 03 — Vector Database (ChromaDB)

> **Free, persistent, local vector storage with metadata filtering and MMR search**

---

## 1. Why ChromaDB?

| Feature | ChromaDB | Pinecone | Qdrant Cloud | Weaviate Cloud |
|---------|----------|----------|-------------|----------------|
| **Cost** | Free (Apache 2.0) | Free tier (1M vectors) | Free tier (1M vectors) | Free tier (1M vectors) |
| **Self-hosted** | ✅ Embedded | ❌ | ✅ Open source | ✅ Open source |
| **Persistent** | ✅ On-disk | ✅ Managed | ✅ Managed | ✅ Managed |
| **Metadata filters** | ✅ Full | ✅ Partial | ✅ Full | ✅ Full |
| **MMR search** | ✅ Built-in | ❌ | ✅ | ✅ |
| **Async** | ✅ | ✅ | ✅ | ✅ |
| **No server needed** | ✅ (in-process) | ❌ | ❌ | ❌ |
| **Max vectors** | Unlimited (local) | 1M (free) | 1M (free) | 1M (free) |

**Verdict:** ChromaDB is the only fully free option that runs embedded with zero infrastructure. No server process, no cloud dependency, unlimited vectors on your hardware.

---

## 2. Installation

```bash
pip install chromadb  # Core package
pip install chromadb-client  # Optional: separate client for remote access
```

### ChromaDB Modes

| Mode | Use Case | Storage |
|------|----------|---------|
| **In-memory** | Testing, prototyping | RAM only (lost on restart) |
| **Persistent (on-disk)** | Production | `./chroma_db/` directory |
| **HTTP client-server** | Multi-process, scaling | Remote Chroma server |

---

## 3. ChromaDB Client Wrapper

```python
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import Optional
import hashlib
import json

class ChromaStore:
    """Production-ready wrapper around ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "rag_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        distance_metric: str = "cosine"
    ):
        self.persist_directory = persist_directory
        
        # Use sentence-transformers locally for free embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Initialize client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=False,           # Prevent accidental data loss
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": distance_metric}  # Default: cosine
        )

    def get_collection_stats(self) -> dict:
        """Return collection statistics."""
        count = self.collection.count()
        return {
            "name": self.collection.name,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_fn._model_name,
        }
```

---

## 4. Embedding Model Strategy

### Free Embedding Models

| Model | Dimensions | Size | Speed (CPU) | Quality | License |
|-------|-----------|------|-------------|---------|---------|
| **all-MiniLM-L6-v2** | 384 | 80MB | ★★★★★ Fast | ★★★★ Good | Apache 2.0 |
| **all-mpnet-base-v2** | 768 | 420MB | ★★★ Medium | ★★★★★ Best | Apache 2.0 |
| **BAAI/bge-small-en-v1.5** | 384 | 33MB | ★★★★★ Fast | ★★★★ Good | MIT |
| **BAAI/bge-base-en-v1.5** | 768 | 130MB | ★★★★ Good | ★★★★★ Best | MIT |
| **nomic-embed-text-v1.5** | 768 | 140MB | ★★★★ Good | ★★★★★ Best | Apache 2.0 |

**Recommended default:** `all-MiniLM-L6-v2` — best speed-to-quality ratio for local RAG.

**For higher accuracy:** `BAAI/bge-base-en-v1.5` — better at domain-specific retrieval.

### Embedding via Ollama (Alternative)

```python
class OllamaEmbeddingFunction:
    """Use Ollama's embedding endpoint (nomic-embed-text)."""

    def __init__(self, model: str = "nomic-embed-text"):
        import httpx
        self.model = model
        self.client = httpx.Client(base_url="http://localhost:11434")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            resp = self.client.post("/api/embeddings", json={
                "model": self.model,
                "prompt": text
            })
            embeddings.append(resp.json()["embedding"])
        return embeddings
```

**Trade-off:** Ollama embeddings run on GPU (faster) but require Ollama to be running. Sentence-transformers run anywhere but use CPU.

---

## 5. Document Ingestion (Add to Vector DB)

```python
def add_document_chunks(
    self,
    chunks: list[str],
    metadatas: list[dict],
    ids: Optional[list[str]] = None
) -> list[str]:
    """
    Add document chunks to the vector store.
    
    Args:
        chunks: List of text chunks
        metadatas: List of metadata dicts (one per chunk)
        ids: Optional custom IDs (auto-generated if None)
    
    Returns:
        List of chunk IDs
    """
    if ids is None:
        ids = []
        for i, chunk in enumerate(chunks):
            # Content-addressed ID for deduplication
            content_hash = hashlib.md5(chunk.encode()).hexdigest()[:12]
            source = metadatas[i].get("source", "unknown")
            ids.append(f"{source}_{content_hash}_{i}")

    # Batch add to ChromaDB
    self.collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    return ids

def delete_document(self, source: str) -> int:
    """Delete all chunks belonging to a document."""
    result = self.collection.delete(
        where={"source": source}
    )
    return len(result) if result else 0
```

### Complete Ingestion Pipeline

```python
def ingest_document(
    self,
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> dict:
    """
    Full ingestion pipeline: read → chunk → embed → store.
    
    Returns ingestion stats.
    """
    from backend.core.chunker import DocumentChunker
    
    # 1. Read file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    source = os.path.basename(file_path)
    
    # 2. Chunk
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, chunk_metadatas = chunker.chunk(content, source=source)
    
    # 3. Add to vector store
    ids = self.add_document_chunks(chunks, chunk_metadatas)
    
    return {
        "source": source,
        "num_chunks": len(chunks),
        "total_chars": len(content),
        "chunk_ids": ids,
    }
```

---

## 6. Retrieval

### Basic Similarity Search

```python
def search(
    self,
    query: str,
    n_results: int = 5,
    where: Optional[dict] = None,
    where_document: Optional[dict] = None
) -> dict:
    """
    Search for similar chunks.
    
    Args:
        query: Search query
        n_results: Number of results to return
        where: Metadata filter (e.g., {"source": "report.pdf"})
        where_document: Document content filter
    
    Returns:
        {"documents": [...], "metadatas": [...], "distances": [...], "ids": [...]}
    """
    results = self.collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        where_document=where_document,
        include=["documents", "metadatas", "distances"]
    )
    return results
```

### MMR Search (Maximum Marginal Relevance)

MMR balances relevance and diversity — essential for comprehensive RAG.

```python
def search_mmr(
    self,
    query: str,
    n_results: int = 5,
    lambda_mult: float = 0.5,
    fetch_k: int = 20,
    where: Optional[dict] = None
) -> dict:
    """
    MMR search: diversity + relevance.
    
    Args:
        lambda_mult: 0 = max diversity, 1 = max relevance
        fetch_k: Number of candidates to consider (higher = better but slower)
    """
    results = self.collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
        # ChromaDB MMR parameters
        # Note: ChromaDB uses internal MMR via the embedding function
    )
    return results
```

> **Note on ChromaDB MMR:** As of ChromaDB v0.5+, MMR is available via the `mmr` parameter in `query()`. Check the latest API for exact syntax. Alternatively, implement custom MMR post-query.

---

## 7. Metadata Filtering

Metadata filtering is crucial for RAG — it lets users search within specific documents, date ranges, or categories.

### Metadata Schema

Define a consistent metadata schema for every chunk:

```python
METADATA_SCHEMA = {
    "source": str,       # Filename (e.g., "report.pdf")
    "page": int,         # Page number (if applicable)
    "chunk_index": int,  # Position in document
    "total_chunks": int, # Total chunks for this document
    "file_type": str,    # "pdf", "txt", "md", "docx"
    "uploaded_at": str,  # ISO 8601 timestamp
    "category": str,     # Optional: user-defined category
    "tags": str,         # Optional: comma-separated tags
}
```

### Filter Examples

```python
# Search only in a specific document
results = store.search("budget forecast", where={"source": "q4-report.pdf"})

# Search within a date range
results = store.search("meeting notes", where={
    "$and": [
        {"uploaded_at": {"$gte": "2025-01-01"}},
        {"uploaded_at": {"$lte": "2025-12-31"}}
    ]
})

# Search by file type
results = store.search("API docs", where={"file_type": "md"})

# Search with tags (if stored as comma-separated string)
# Requires preprocessing or using $contains
results = store.search("Python code", where={"tags": {"$contains": "python"}})
```

---

## 8. Collection Management

### Multiple Collections

For multi-user or multi-domain setups, use separate collections:

```python
class MultiCollectionStore:
    """Manage multiple collections (one per user/project)."""

    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(path=persist_directory)

    def get_user_collection(self, user_id: str, embedding_fn):
        return self.client.get_or_create_collection(
            name=f"user_{user_id}",
            embedding_function=embedding_fn
        )

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, name: str):
        self.client.delete_collection(name)
```

### Updating and Deleting

```python
# Update metadata for specific chunks
collection.update(
    ids=["chunk_001", "chunk_002"],
    metadatas=[{"source": "renamed.pdf"}, {"source": "renamed.pdf"}]
)

# Delete chunks
collection.delete(ids=["chunk_001"])
collection.delete(where={"source": "outdated.pdf"})

# Reset entire collection (careful!)
collection.delete(where={})  # Deletes all but keeps collection

# Drop and recreate
client.delete_collection("rag_documents")
```

---

## 9. Performance Optimization

### HNSW Index Tuning

ChromaDB uses HNSW (Hierarchical Navigable Small World) indexing. Key parameters:

```python
collection = client.create_collection(
    name="optimized",
    metadata={
        "hnsw:space": "cosine",         # Distance metric
        "hnsw:construction_ef": 200,    # Higher = better recall, slower build
        "hnsw:M": 32,                   # Higher = better recall, more memory
        "hnsw:search_ef": 100,          # Search depth
        "hnsw:num_threads": 4,          # Parallel index building
    }
)
```

| Parameter | Low | High | Default |
|-----------|-----|------|---------|
| `hnsw:construction_ef` | 50 (fast build) | 400 (best recall) | 100 |
| `hnsw:M` | 8 (low memory) | 64 (best recall) | 16 |
| `hnsw:search_ef` | 50 (fast search) | 200 (best recall) | 10 |

### Batch Processing (Faster Ingestion)

```python
def add_documents_batch(
    self,
    documents: list[str],
    metadatas: list[dict],
    batch_size: int = 100
) -> list[str]:
    """Add documents in batches to avoid memory spikes."""
    all_ids = []
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        
        batch_ids = [
            hashlib.md5(doc.encode()).hexdigest()[:12]
            for doc in batch_docs
        ]
        
        self.collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        all_ids.extend(batch_ids)
    
    return all_ids
```

### Persistent Storage Size Estimate

```
1 chunk ≈ 512 tokens ≈ 2,000 characters ≈ 2KB text
1 embedding (384-dim float32) ≈ 1.5KB
1 metadata entry ≈ 0.5KB
Total per chunk ≈ 4KB

10,000 chunks ≈ 40MB
100,000 chunks ≈ 400MB
1,000,000 chunks ≈ 4GB
```

---

## 10. Backup and Migration

### Backup

```python
import shutil
import json

def backup_collection(persist_dir: str, backup_path: str):
    """Backup entire ChromaDB persistent store."""
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(persist_dir, backup_path)
    print(f"Backup saved to {backup_path}")

# Also export metadata as JSON for inspection
def export_metadata(persist_dir: str, output_path: str):
    """Export all chunks and metadata as JSON."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("rag_documents")
    all_data = collection.get(include=["documents", "metadatas"])
    
    with open(output_path, "w") as f:
        json.dump({
            "ids": all_data["ids"],
            "documents": all_data["documents"],
            "metadatas": all_data["metadatas"],
        }, f, indent=2)
```

### Migration to Another Vector DB

Since the data export is just JSON + text files, migration to Qdrant, Weaviate, or Pinecone is straightforward:

```python
def migrate_to_qdrant(export_path: str, qdrant_url: str, collection_name: str):
    """Migrate from ChromaDB export to Qdrant."""
    from qdrant_client import QdrantClient
    import json
    
    with open(export_path) as f:
        data = json.load(f)
    
    client = QdrantClient(url=qdrant_url)
    # Create collection, upsert points...
```

---

## 11. Testing and Validation

```python
def test_vector_store():
    """Quick smoke test for the vector store."""
    store = ChromaStore(persist_directory="./test_chroma")
    
    # Add test chunks
    store.add_document_chunks(
        chunks=["The capital of France is Paris.",
                "The Eiffel Tower is in Paris.",
                "Python is a programming language."],
        metadatas=[{"source": "test.txt", "chunk_index": i}
                   for i in range(3)]
    )
    
    # Search
    results = store.search("What is the capital of France?", n_results=2)
    assert len(results["documents"][0]) == 2
    assert "Paris" in results["documents"][0][0]
    
    # Filter
    results = store.search("Python", where={"source": "test.txt"})
    assert len(results["documents"][0]) >= 1
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_chroma")
    
    print("✅ All vector store tests passed!")
```

---

## 12. Checklist

- [ ] ChromaDB installed (`pip install chromadb`)
- [ ] Sentence-transformers model downloaded (auto on first use)
- [ ] `PersistentClient` configured with correct path
- [ ] Telemetry disabled (`anonymized_telemetry=False`)
- [ ] Metadata schema defined and consistent
- [ ] Batch processing configured for large documents
- [ ] HNSW parameters tuned for your workload
- [ ] Backup strategy implemented
- [ ] Cleanup/deletion functions tested
