"""ChromaDB wrapper for vector storage and retrieval.

Provides persistent client setup, document ingestion with batch support,
similarity and MMR search, metadata filtering, collection management,
and cleanup utilities. Uses sentence-transformers locally for free embeddings.
"""

import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from backend.config import settings as app_settings


class ChromaStore:
    """Production-ready wrapper around ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "",
        collection_name: str = "",
        embedding_model: str = "",
    ):
        persist_directory = persist_directory or app_settings.chroma_persist_path
        collection_name = collection_name or app_settings.chroma_collection
        embedding_model = embedding_model or app_settings.embedding_model

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingestion ──

    def add_chunks(
        self,
        chunks: list[str],
        metadatas: list[dict],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add document chunks to the vector store.

        Args:
            chunks: List of text chunks.
            metadatas: One metadata dict per chunk.
            ids: Optional custom IDs. Auto-generated from content hash if omitted.

        Returns:
            List of chunk IDs.
        """
        if ids is None:
            ids = []
            for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
                h = hashlib.md5(chunk.encode()).hexdigest()[:12]
                source = meta.get("source", "unknown")
                ids.append(f"{source}_{h}_{i}")

        # Batch to avoid memory spikes
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                documents=chunks[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )
        return ids

    def delete_document(self, source: str) -> int:
        """Delete all chunks belonging to a document by source name."""
        result = self.collection.delete(where={"source": source})
        return len(result) if result else 0

    # ── Retrieval ──

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """Search for similar chunks.

        Returns dict with keys: documents, metadatas, distances, ids.
        """
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    # ── Collection Info ──

    def get_stats(self) -> dict:
        return {
            "name": self.collection.name,
            "total_chunks": self.collection.count(),
        }

    def count(self) -> int:
        return self.collection.count()

    def reset_collection(self):
        """Delete all data but keep the collection."""
        # ChromaDB requires a valid where clause with an operator
        total = self.collection.count()
        if total == 0:
            return
        # Get all IDs and delete them individually
        all_data = self.collection.get(limit=total)
        if all_data and all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])
