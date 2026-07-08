"""File storage management on disk.

Organises uploaded files by date to keep the directory flat.
Provides MD5 hash deduplication, deletion, and usage reporting.
"""

import hashlib
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Optional


class FileStore:
    """Manages file storage on disk under a configurable base path."""

    def __init__(self, base_path: str = "./data/documents"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ── Write ──

    def store(self, file_bytes: bytes, original_filename: str) -> str:
        """Persist an uploaded file and return its absolute path.

        Organises files as:  base_path / YYYY-MM-DD / unique_name.ext
        """
        date_dir = self.base_path / str(date.today())
        date_dir.mkdir(exist_ok=True)

        _, ext = os.path.splitext(original_filename)
        unique_name = f"{uuid.uuid4().hex[:8]}{ext}"
        file_path = date_dir / unique_name

        file_path.write_bytes(file_bytes)
        return str(file_path.resolve())

    # ── Read ──

    def read_text(self, file_path: str) -> Optional[str]:
        """Read a text file. Returns None for binary files."""
        path = Path(file_path)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            return None

    def read_bytes(self, file_path: str) -> Optional[bytes]:
        """Read raw bytes from a file."""
        path = Path(file_path)
        if not path.exists():
            return None
        return path.read_bytes()

    # ── Hash ──

    @staticmethod
    def md5(file_path: str) -> str:
        """Compute MD5 hash of a file."""
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def md5_bytes(data: bytes) -> str:
        """Compute MD5 hash of raw bytes."""
        return hashlib.md5(data).hexdigest()

    # ── Delete ──

    def delete(self, file_path: str) -> bool:
        """Remove a file from disk. Returns True if deleted."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return True
        except OSError:
            pass
        return False

    # ── Stats ──

    def usage(self) -> dict:
        """Return storage statistics for the entire file store."""
        total_bytes = 0
        file_count = 0
        for f in self.base_path.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
                file_count += 1
        return {
            "total_files": file_count,
            "total_size_bytes": total_bytes,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "base_path": str(self.base_path.resolve()),
        }
