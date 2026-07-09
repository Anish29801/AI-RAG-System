"""Start the API server for testing."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["RAG_DEBUG"] = "false"

from backend.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
