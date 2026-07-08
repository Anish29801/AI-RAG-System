"""Cross-encoder reranker for improving retrieval precision.

Models:
  - BAAI/bge-reranker-v2-m3 (best quality, ~1.2 GB, ~500ms/query)
  - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, ~80 MB, ~100ms/query)

Typically adds 200-500ms latency but significantly boosts top-K quality.
"""

from typing import Optional
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker — re-scores retrieved chunks by relevance."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[dict]:
        """Rerank documents by relevance to the query.

        Args:
            query: User query string.
            documents: List of document chunk texts.
            top_k: Number of top results to return.
            batch_size: Prediction batch size.

        Returns:
            List of {"content": str, "relevance_score": float}, sorted by score descending.
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=batch_size)

        scored = list(zip(documents, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {"content": doc, "relevance_score": round(float(score), 4)}
            for doc, score in scored[:top_k]
        ]
