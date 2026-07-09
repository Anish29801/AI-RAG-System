"""Application configuration via Pydantic Settings.

Loads from .env file and environment variables with RAG_ prefix.
All config is validated at import time to catch misconfiguration early.
"""

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    # ── LLM ──
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    temperature: float = 0.1
    top_p: float = 0.9
    llm_top_k: int = 40

    # ── Vector Store ──
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_path: str = "./data/chroma_db"
    chroma_collection: str = "rag_documents"

    # ── PDS ──
    pds_db_path: str = "./data/pds.db"
    documents_path: str = "./data/documents"

    # ── RAG Pipeline ──
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    n_results: int = 10
    use_reranker: bool = False
    max_context_tokens: int = 5000

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # ── Limits ──
    max_file_size_mb: int = 50
    max_concurrent_llm: int = 1

    # ── Allowed file types for upload ──
    allowed_file_types: list[str] = [
        ".txt", ".md", ".py", ".js", ".ts", ".csv", ".json",
        ".pdf", ".docx", ".html", ".xml", ".yaml", ".yml",
    ]

    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        frozen = False


# Singleton for fast access across modules
settings = AppConfig()
