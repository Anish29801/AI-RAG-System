# 04 — RAG Pipeline

> **Full Retrieval-Augmented Generation pipeline design — chunking, retrieval, reranking, context assembly, citation**

---

## 1. Pipeline Architecture

```
                        ┌──────────────────┐
                        │   User Query     │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Query Transform  │
                        │ (rewrite/expand) │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
            ┌──────────────┐         ┌──────────────┐
            │ Dense        │         │ Keyword      │
            │ (Vector)     │         │ (BM25)       │
            │ Search       │         │ Search       │
            └──────┬───────┘         └──────┬───────┘
                   │                        │
                   └──────────┬─────────────┘
                              ▼
                     ┌────────────────┐
                     │ Fusion         │
                     │ (RRF)          │
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │ Reranker       │
                     │ (Cross-encoder)│
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │ Context        │
                     │ Assembly       │
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │ LLM Generation │
                     │ (Ollama)       │
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │ Response +     │
                     │ Citations      │
                     └────────────────┘
```

---

## 2. Document Chunking Strategy

Chunking is the most impactful design decision in RAG. Bad chunks = bad retrieval = bad answers.

### Chunking Strategies

| Strategy | Best For | Chunk Size | Overlap | Quality |
|----------|----------|-----------|---------|---------|
| **Recursive Character Split** | Mixed content (default) | 512-1024 chars | 64-128 chars | ★★★★ |
| **Semantic Chunking** | Coherent paragraphs | Variable | None | ★★★★★ |
| **Token Split** | Code, structured text | 256-512 tokens | 32 tokens | ★★★ |
| **Markdown/HTML Split** | Documentation, articles | By heading | None | ★★★★ |
| **Sentence Split** | Legal, academic | By sentence | 1-2 sentences | ★★★ |

### Recursive Character Splitter (Default)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_default_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> RecursiveCharacterTextSplitter:
    """Get a recursive character text splitter tuned for RAG."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Priority order
    )
```

### Semantic Chunking (Best Quality)

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticChunker:
    """
    Chunks text by detecting topic boundaries using embedding similarity.
    More computationally expensive but produces better chunks.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.3,  # Lower = more chunks
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return sentences
        
        # Get embeddings for each sentence
        embeddings = self.model.encode(sentences)
        
        # Calculate cosine similarity between adjacent sentences
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            current_size = len(" ".join(current_chunk))
            
            if sim < self.threshold and current_size >= self.min_chunk_size:
                # Topic boundary detected
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            elif current_size >= self.max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences (simple heuristic)."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
```

### Chunking by Document Type

```python
class DocumentChunker:
    """Select chunking strategy based on document type."""
    
    STRATEGIES = {
        ".md": "markdown",
        ".txt": "recursive",
        ".pdf": "recursive",  # After text extraction
        ".py": "token",
        ".js": "token",
        ".html": "markdown",
        ".csv": "sentence",
    }
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        default_strategy: str = "recursive"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_strategy = default_strategy
    
    def chunk(
        self,
        content: str,
        source: str = "",
        file_type: str = ""
    ) -> tuple[list[str], list[dict]]:
        """Chunk content and return (chunks, metadatas)."""
        if not file_type:
            _, ext = os.path.splitext(source)
            file_type = self.STRATEGIES.get(ext, self.default_strategy)
        
        if file_type == "markdown":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
            )
        else:  # recursive
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        
        chunks = splitter.split_text(content)
        
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "file_type": file_type,
            })
        
        return chunks, metadatas
```

---

## 3. Hybrid Search (Dense + Sparse)

Dense search (vector) captures semantics. Sparse search (keyword) captures exact terms. Together, they beat either alone.

### BM25 (Sparse) Implementation

```python
import math
from collections import Counter
from typing import List

class BM25Okapi:
    """Pure Python BM25 implementation (no external dependencies)."""
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc.split()) for doc in corpus) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self._initialize(corpus)
    
    def _initialize(self, corpus: List[str]):
        for document in corpus:
            freq = Counter(document.split())
            self.doc_freqs.append(freq)
            for word in freq:
                self.idf[word] = self.idf.get(word, 0) + 1
        
        # Calculate IDF
        for word, freq in self.idf.items():
            self.idf[word] = math.log(
                1 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
            )
    
    def get_scores(self, query: str) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q_word in query.split():
            if q_word not in self.idf:
                continue
            idf = self.idf[q_word]
            for i, doc_freq in enumerate(self.doc_freqs):
                doc_len = sum(doc_freq.values())
                tf = doc_freq.get(q_word, 0)
                score = idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                )
                scores[i] += score
        return scores
```

### Reciprocal Rank Fusion (RRF)

Combines dense and sparse results:

```python
def rrf_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_n: int = 10
) -> list[dict]:
    """
    Reciprocal Rank Fusion — combine two ranked result lists.
    
    Args:
        dense_results: Results from vector search
        sparse_results: Results from BM25 search
        k: RRF constant (typically 60)
        top_n: Final number of results
    
    Returns:
        Fused and re-ranked results
    """
    scores = {}
    
    def rank_score(rank: int, k: int) -> float:
        return 1.0 / (k + rank)
    
    for rank, result in enumerate(dense_results, 1):
        doc_id = result.get("id", result.get("content", ""))
        scores[doc_id] = scores.get(doc_id, 0) + rank_score(rank, k)
    
    for rank, result in enumerate(sparse_results, 1):
        doc_id = result.get("id", result.get("content", ""))
        scores[doc_id] = scores.get(doc_id, 0) + rank_score(rank, k)
    
    # Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return [{"content": doc_id, "score": score} for doc_id, score in ranked]
```

---

## 4. Reranking with Cross-Encoder

The initial retrieval (bi-encoder) is fast but imprecise. A cross-encoder reranker significantly improves final quality.

### Why Rerank?

```
Bi-encoder:   "weather in Tokyo" → ["Tokyo weather", "Tokyo travel", "Sushi in Tokyo"]
Cross-encoder: "weather in Tokyo" → ["Tokyo weather" (0.95), "Tokyo travel" (0.12), ...]
```

### Free Reranker Options

| Model | Size | Speed | Quality | License |
|-------|------|-------|---------|---------|
| **BAAI/bge-reranker-v2-m3** | 1.2GB | ~500ms/query | ★★★★★ Best | MIT |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | 80MB | ~100ms/query | ★★★★ Good | Apache 2.0 |
| **ms-marco-TinyBERT-L-2-v2** | 27MB | ~50ms/query | ★★★ Fair | Apache 2.0 |

### Reranker Implementation

```python
from sentence_transformers import CrossEncoder
from typing import List

class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu"
    ):
        self.model = CrossEncoder(model_name, device=device)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        batch_size: int = 32
    ) -> List[dict]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: User query
            documents: List of document texts
            top_k: Number of results to return
        
        Returns:
            [{"content": str, "relevance_score": float}, ...]
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=batch_size)
        
        # Pair documents with scores and sort
        scored = list(zip(documents, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {"content": doc, "relevance_score": score}
            for doc, score in scored[:top_k]
        ]
```

### When to Skip Reranking

```
If latency budget < 500ms:  → Skip reranker (use only vector search)
If latency budget < 2000ms: → Use MiniLM reranker (fast)
If latency budget > 2000ms: → Use BGE reranker (best quality)
```

---

## 5. Query Transformation

Users rarely ask perfectly formed questions. Transform the query before retrieval.

### Query Rewriting

```python
import re

class QueryTransformer:
    """Transform user queries for better retrieval."""
    
    @staticmethod
    def expand_acronyms(query: str) -> str:
        """Expand common acronyms for better matching."""
        acronyms = {
            "RAG": "Retrieval Augmented Generation",
            "LLM": "Large Language Model",
            "AI": "Artificial Intelligence",
            "PDS": "Personal Data Store",
            "API": "Application Programming Interface",
        }
        for acro, full in acronyms.items():
            # Only expand standalone acronyms
            query = re.sub(rf'\b{acro}\b', f"{acro} ({full})", query)
        return query
    
    @staticmethod
    def extract_queries(query: str) -> list[str]:
        """Extract sub-questions from a compound query."""
        # Split on question marks, commas, "and"
        sub_queries = re.split(r'[?]|\s+and\s+|\s*,\s*', query)
        return [q.strip() for q in sub_queries if q.strip()]
    
    @staticmethod
    def generate_hypothetical_answer(query: str, llm_client) -> str:
        """
        HyDE (Hypothetical Document Embedding):
        Generate a hypothetical answer, then search for chunks similar to it.
        Improves retrieval for abstract/complex queries.
        """
        hyde_prompt = (
            f"Generate a short, factual paragraph that would answer this question: {query}\n"
            f"Paragraph:"
        )
        return llm_client.generate(hyde_prompt, temperature=0.7)
```

### Multi-Query Retrieval

```python
async def multi_query_retrieval(
    query: str,
    vector_store,
    llm_client,
    n_queries: int = 3,
    n_results_per_query: int = 3
) -> list[dict]:
    """
    Generate multiple query variations and aggregate results.
    """
    # Generate query variations
    prompt = f"""
    Generate {n_queries} different phrasings of this question for searching.
    Each on a new line. Make them semantically different.
    
    Original: {query}
    
    Variations:"""
    
    variations_text = await llm_client.generate(prompt, temperature=0.8)
    variations = [q.strip() for q in variations_text.split('\n') if q.strip()]
    variations.append(query)  # Include original
    
    # Search with each variation
    all_results = []
    seen_ids = set()
    
    for q in variations:
        results = vector_store.search(q, n_results=n_results_per_query)
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            doc_id = f"{meta.get('source', '')}_{doc[:50]}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append({
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # Convert distance to similarity
                })
    
    # Sort by score and return top-k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results
```

---

## 6. Context Assembly

Assembling the final context for the LLM is critical for quality.

```python
class ContextAssembler:
    """Assemble retrieved chunks into an LLM-ready context."""
    
    def __init__(
        self,
        max_tokens: int = 5000,
        token_overhead: int = 200,  # Formatting overhead
    ):
        self.max_tokens = max_tokens
        self.token_overhead = token_overhead
    
    def assemble(
        self,
        chunks: list[dict],
        include_sources: bool = True,
        deduplicate: bool = True
    ) -> str:
        """
        Build a formatted context block from retrieved chunks.
        
        Args:
            chunks: List of {"content": str, "metadata": dict, "score": float}
            include_sources: Whether to include source citations
            deduplicate: Remove near-duplicate chunks
        
        Returns:
            Formatted context string
        """
        if deduplicate:
            chunks = self._deduplicate(chunks)
        
        # Sort by relevance (highest first)
        chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        context_parts = []
        total_chars = 0
        
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source", "unknown") if include_sources else ""
            page = chunk["metadata"].get("page", "")
            
            header = f"[Source: {source}"
            if page:
                header += f", p. {page}"
            header += "]"
            
            entry = f"{header}\n{chunk['content']}\n"
            
            # Check token budget (rough: 4 chars ≈ 1 token)
            estimated_tokens = len(entry) / 4
            if total_chars + estimated_tokens > self.max_tokens:
                break
            
            context_parts.append(entry)
            total_chars += estimated_tokens
        
        return "\n---\n".join(context_parts)
    
    def _deduplicate(self, chunks: list[dict]) -> list[dict]:
        """Remove near-duplicate chunks by content overlap."""
        seen = set()
        unique = []
        for chunk in chunks:
            # Use first 100 chars as fingerprint
            fingerprint = chunk["content"][:100]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(chunk)
        return unique
```

---

## 7. Complete RAG Pipeline Orchestrator

```python
class RAGPipeline:
    """End-to-end RAG pipeline orchestrator."""

    def __init__(
        self,
        vector_store,
        llm_client,
        reranker: Optional[Reranker] = None,
        context_assembler: Optional[ContextAssembler] = None,
        use_hyde: bool = False,
        use_multi_query: bool = False,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.reranker = reranker
        self.context_assembler = context_assembler or ContextAssembler()
        self.use_hyde = use_hyde
        self.use_multi_query = use_multi_query
    
    async def answer(
        self,
        query: str,
        n_results: int = 10,
        top_k: int = 5,
        where: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> dict:
        """
        Full RAG pipeline: transform → retrieve → rerank → assemble → generate.
        
        Returns:
            {
                "answer": str,
                "sources": [{"source": str, "page": str, "content": str, "score": float}],
                "tokens_used": int,
                "latency_ms": float
            }
        """
        import time
        start = time.time()
        
        # 1. Query Transformation
        transformed_query = query
        if self.use_hyde:
            hyde_doc = QueryTransformer.generate_hypothetical_answer(query, self.llm_client)
            transformed_query = f"{query}\n\nHypothetical: {hyde_doc}"
        
        # 2. Retrieval
        if self.use_multi_query:
            chunks = await multi_query_retrieval(
                query, self.vector_store, self.llm_client,
                n_results_per_query=max(3, n_results // 2)
            )
        else:
            # Single query retrieval
            results = self.vector_store.search(
                transformed_query,
                n_results=n_results,
                where=where
            )
            chunks = [
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],
                }
                for i in range(len(results["documents"][0]))
            ]
        
        # 3. Reranking (optional)
        if self.reranker:
            doc_texts = [c["content"] for c in chunks]
            reranked = self.reranker.rerank(query, doc_texts, top_k=top_k)
            
            # Map scores back to original metadata
            scored_chunks = []
            for r in reranked:
                original = next(
                    c for c in chunks if c["content"] == r["content"]
                )
                scored_chunks.append({
                    **original,
                    "score": r["relevance_score"],
                })
            chunks = scored_chunks
        else:
            chunks = chunks[:top_k]
        
        # 4. Context Assembly
        context = self.context_assembler.assemble(chunks)
        
        # 5. Build prompt
        system = system_prompt or DEFAULT_RAG_SYSTEM_PROMPT
        prompt = build_rag_prompt(query, chunks)
        
        # 6. Generate
        answer = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=system,
            stream=stream,
        )
        
        latency = (time.time() - start) * 1000
        
        # 7. Extract sources for citation
        sources = [
            {
                "source": c["metadata"].get("source", "unknown"),
                "page": c["metadata"].get("page", ""),
                "content_preview": c["content"][:200],
                "score": round(c["score"], 4),
            }
            for c in chunks[:top_k]
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": round(latency, 0),
            "query": query,
        }


DEFAULT_RAG_SYSTEM_PROMPT = """You are a precise RAG assistant. Answer using ONLY the provided context.

Rules:
- Never use your internal knowledge when context is provided.
- If the context doesn't contain the answer, say so.
- Cite sources as [Source: filename].
- Be concise and factual."""


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Build the final prompt with context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        context_parts.append(
            f"[Document {i}: {source}]\n{chunk['content']}"
        )
    
    return (
        "Answer the question using ONLY the context below.\n\n"
        "--- CONTEXT ---\n"
        + "\n\n".join(context_parts) +
        "\n--- END CONTEXT ---\n\n"
        f"Question: {query}\n\n"
        "Answer (cite sources as [Source: filename]):"
    )
```

---

## 8. Quality Metrics

```python
def evaluate_rag_pipeline(
    pipeline: RAGPipeline,
    test_queries: list[dict],
    embedding_model: str = "all-MiniLM-L6-v2"
) -> dict:
    """
    Evaluate RAG pipeline quality.
    
    test_queries: [{"query": str, "relevant_chunks": [str], "expected_answer": str}]
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer(embedding_model)
    
    metrics = {
        "hit_rate": [],
        "mrr": [],       # Mean Reciprocal Rank
        "answer_relevance": [],
    }
    
    for test in test_queries:
        result = pipeline.answer(test["query"])
        retrieved = [s["source"] for s in result["sources"]]
        relevant = test["relevant_chunks"]
        
        # Hit Rate
        hits = sum(1 for r in retrieved if r in relevant)
        metrics["hit_rate"].append(hits / len(relevant) if relevant else 0)
        
        # MRR
        for rank, r in enumerate(retrieved, 1):
            if r in relevant:
                metrics["mrr"].append(1.0 / rank)
                break
        else:
            metrics["mrr"].append(0.0)
        
        # Answer relevance (cosine sim between expected and generated)
        exp_emb = model.encode(test["expected_answer"])
        gen_emb = model.encode(result["answer"])
        sim = np.dot(exp_emb, gen_emb) / (
            np.linalg.norm(exp_emb) * np.linalg.norm(gen_emb)
        )
        metrics["answer_relevance"].append(float(sim))
    
    return {
        "avg_hit_rate": np.mean(metrics["hit_rate"]),
        "avg_mrr": np.mean(metrics["mrr"]),
        "avg_answer_relevance": np.mean(metrics["answer_relevance"]),
        "num_queries": len(test_queries),
    }
```

---

## 9. Pipeline Checklist

- [ ] Default chunker configured (RecursiveCharacter, 512/64)
- [ ] Metadata attached to every chunk (source, chunk_index, etc.)
- [ ] Vector search tested with basic query
- [ ] Metadata filtering working (search within specific doc)
- [ ] Hybrid search (dense + BM25) implemented (optional)
- [ ] Reranker configured with fallback (skip if unavailable)
- [ ] Query transformation (rewriting, multi-query) done
- [ ] Context assembler respects token budget
- [ ] Build RAG prompt handles empty context gracefully
- [ ] Streaming responses work end-to-end
- [ ] Sources/citations returned with every answer
- [ ] Pipeline handles errors gracefully (no crash on empty results)
- [ ] Latency measured and optimized (< 5s total for typical query)
