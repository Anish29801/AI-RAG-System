"""Embedding generation using free sentence-transformers models.

Default model: all-MiniLM-L6-v2 (384-dim, 80 MB, fast CPU inference).
Supports batch processing and fallback to Ollama embedding API.
"""

import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

from backend.config import settings


class Embedder:
    """Generate embeddings for text using a local sentence-transformers model.

    Usage:
        embedder = Embedder()
        vec = embedder.encode("some text")          # single vector
        vecs = embedder.encode_many(["a", "b"])     # batch
    """

    def __init__(self, model_name: str = "", device: str = "cpu"):
        model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> list[float]:
        """Embed a single text string. Returns a 384/768-dim float list."""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def encode_many(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts. Normalised for cosine similarity."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return str(self.model)
