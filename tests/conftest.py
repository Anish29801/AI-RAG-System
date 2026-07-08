"""Shared fixtures and utilities for all test batches.

Usage:
    pytest tests/ --fixtures   # list all available fixtures
    pytest tests/test_foundation.py -v
"""

import os
import sys
import tempfile
import shutil
from typing import Generator, AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import AppConfig
from backend.pds.repository import PDSRepository
from backend.pds.file_store import FileStore
from backend.vector_store.chroma_client import ChromaStore
from backend.core.llm_client import LLMClient
from backend.core.rag_pipeline import RAGPipeline
try:
    from backend.core.embedder import Embedder
except ImportError:
    Embedder = None  # type: ignore
from backend.core.chunker import DocumentChunker


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory that auto-cleans up."""
    d = tempfile.mkdtemp(prefix="rag_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_config() -> AppConfig:
    """Return default AppConfig (reads from .env if present)."""
    return AppConfig()


@pytest.fixture
def file_store(temp_dir: str) -> FileStore:
    """FileStore backed by a temp directory."""
    return FileStore(base_path=os.path.join(temp_dir, "docs"))


@pytest.fixture
def chroma_store(temp_dir: str) -> Generator[ChromaStore, None, None]:
    """ChromaStore backed by a temp directory. Auto-cleans up."""
    store = ChromaStore(
        persist_directory=os.path.join(temp_dir, "chroma"),
        collection_name="test_coll",
    )
    yield store
    # No explicit cleanup needed — temp_dir removal handles it


@pytest_asyncio.fixture
async def pds_repo(temp_dir: str) -> AsyncGenerator[PDSRepository, None]:
    """PDSRepository backed by a temp SQLite database."""
    db_path = os.path.join(temp_dir, "test.db")
    repo = PDSRepository(db_path=db_path)
    await repo.initialize()
    yield repo
    await repo.close()


@pytest.fixture
def chunker() -> DocumentChunker:
    """Default DocumentChunker with 512/64 chunking."""
    return DocumentChunker(chunk_size=512, chunk_overlap=64)


@pytest.fixture
def small_chunker() -> DocumentChunker:
    """Small chunker for testing overlap/edge cases."""
    return DocumentChunker(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def embedder():
    """Sentence-transformers embedder (CPU, ~80 MB). Skip if not installed."""
    if Embedder is None:
        pytest.skip("sentence-transformers not installed")
    return Embedder()


# ═══════════════════════════════════════════════════════════════
# Mocks
# ═══════════════════════════════════════════════════════════════


class MockLLMClient:
    """A fully mocked LLM client returning canned responses.

    Tracks call count for asserting LLM interaction patterns.
    """

    def __init__(self, response: str = "Mock response based on provided context."):
        self.response = response
        self.generated_count = 0
        self.last_prompt = ""
        self.last_system_prompt = ""

    async def generate(
        self, prompt: str, system_prompt: str = "", stream: bool = False
    ):
        self.generated_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        if stream:
            async def _token_gen():
                for word in self.response.split():
                    yield word + " "
            return _token_gen()
        return self.response

    async def chat(self, messages: list[dict], stream: bool = False):
        self.generated_count += 1
        return self.response

    async def is_available(self) -> bool:
        return True

    async def close(self):
        pass


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Mock LLM client that returns canned responses."""
    return MockLLMClient()


@pytest.fixture
def rag_pipeline(chroma_store, mock_llm) -> RAGPipeline:
    """RAGPipeline wired to temp ChromaStore + mock LLM."""
    return RAGPipeline(vector_store=chroma_store, llm_client=mock_llm)


# ═══════════════════════════════════════════════════════════════
# Sample data
# ═══════════════════════════════════════════════════════════════

SAMPLE_TEXT = (
    "The capital of France is Paris. The Eiffel Tower is located in Paris. "
    "France is known for its cuisine, art, and culture. "
    "The French Revolution began in 1789. "
    "Paris is also called the City of Light."
)

SAMPLE_MARKDOWN = (
    "# Meeting Notes\n\n"
    "Discussed Q3 budget allocation.\n\n"
    "## Action Items\n\n"
    "- Finalise quarterly report\n"
    "- Send invoice to client\n"
    "- Schedule follow-up meeting\n\n"
    "## Budget Summary\n\n"
    "Total: $50,000\n"
    "Marketing: $20,000\n"
    "Engineering: $30,000"
)

SAMPLE_CODE = (
    "def hello():\n"
    "    print('Hello, world!')\n\n"
    "def add(a, b):\n"
    "    return a + b\n\n"
    "class Calculator:\n"
    "    def multiply(self, x, y):\n"
    "        return x * y\n"
)
