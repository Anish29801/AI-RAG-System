"""End-to-end RAG pipeline orchestrator.

Coordinates: query transformation → retrieval → reranking →
context assembly → LLM generation → response with citations.
"""

import time
from typing import Optional

from backend.config import settings
from backend.core.llm_client import LLMClient
from backend.vector_store.chroma_client import ChromaStore

# Optional — silences import error when reranker is not installed
try:
    from backend.core.reranker import Reranker
except ImportError:
    Reranker = None  # type: ignore


# ── Prompt Templates ──

SYSTEM_PROMPT_RAG = (
    "You are a precise RAG assistant. Answer the user's question using "
    "ONLY the context provided below. If the context does not contain "
    "enough information, say so clearly. Cite sources as [Source: filename]."
)

CONTEXT_TEMPLATE = (
    "Answer the question using ONLY the context below.\n\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "--- END CONTEXT ---\n\n"
    "Question: {query}\n\n"
    "Answer (cite sources as [Source: filename]):"
)


# ── Pipeline ──


class RAGPipeline:
    """End-to-end RAG pipeline orchestrator."""

    def __init__(
        self,
        vector_store: ChromaStore,
        llm_client: LLMClient,
        reranker: Optional["Reranker"] = None,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.reranker = reranker

    async def answer(
        self,
        query: str,
        n_results: int = 10,
        top_k: int = 5,
        where: Optional[dict] = None,
        stream: bool = False,
    ) -> dict:
        """Run the full RAG pipeline.

        Args:
            query: User's question.
            n_results: Number of chunks to retrieve from vector store.
            top_k: Number of chunks after reranking (or top-N truncation).
            where: Optional metadata filter for vector search.
            stream: Whether to stream the LLM response.

        Returns:
            dict with keys: answer, sources, latency_ms, query.
            If stream=True, answer is an async generator of tokens.
        """
        start = time.time()

        # 1. Retrieve from vector store
        results = self.vector_store.search(
            query, n_results=n_results, where=where,
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            })

        # 2. Optional reranking
        if self.reranker and chunks:
            doc_texts = [c["content"] for c in chunks]
            reranked = self.reranker.rerank(query, doc_texts, top_k=top_k)
            # Map scores back to original metadata
            text_map = {c["content"]: c for c in chunks}
            chunks = []
            for r in reranked:
                original = text_map.get(r["content"])
                if original:
                    chunks.append({**original, "score": r["relevance_score"]})
        else:
            chunks = chunks[:top_k]

        # 3. Assemble context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source", "unknown")
            context_parts.append(f"[Document {i}: {source}]\n{chunk['content']}")

        context = "\n\n".join(context_parts)

        # 4. Build prompt
        prompt = CONTEXT_TEMPLATE.format(context=context, query=query)

        # 5. Generate answer
        answer = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_RAG,
            stream=stream,
        )

        latency = round((time.time() - start) * 1000)

        # 6. Extract sources for citation
        sources = [
            {
                "source": c["metadata"].get("source", "unknown"),
                "page": c["metadata"].get("page", ""),
                "content_preview": c["content"][:200],
                "score": round(float(c["score"]), 4),
            }
            for c in chunks[:top_k]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency,
            "query": query,
        }
