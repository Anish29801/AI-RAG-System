"""Persistent storage for LLM settings and change history.

Saves settings to a JSON file so they survive server restarts.
Logs every settings change with a timestamp for the dashboard chart.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional


class SettingsStore:
    """Persists LLM settings (model, temperature, top_p, top_k) to a JSON file."""

    def __init__(self, file_path: str = "./data/llm_settings.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[dict] = None

    def load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                self._cache = data
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def save(self, settings: dict):
        self._cache = settings
        self.file_path.write_text(
            json.dumps(settings, indent=2, default=str),
            encoding="utf-8",
        )


class HistoryStore:
    """Stores a rolling window of LLM parameter changes in a JSON file."""

    MAX_ENTRIES = 100

    def __init__(self, file_path: str = "./data/param_history.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[list] = None

    def _read(self) -> list:
        if self._cache is not None:
            return self._cache
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                self._cache = data if isinstance(data, list) else []
                return self._cache
            except (json.JSONDecodeError, OSError):
                pass
        self._cache = []
        return self._cache

    def _write(self, entries: list):
        self._cache = entries
        self.file_path.write_text(
            json.dumps(entries, indent=2, default=str),
            encoding="utf-8",
        )

    def record(self, temperature: float, top_p: float, top_k: int, model: str):
        entries = self._read()
        entries.append({
            "timestamp": time.time(),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "model": model,
        })
        # Trim to max entries
        if len(entries) > self.MAX_ENTRIES:
            entries = entries[-self.MAX_ENTRIES:]
        self._write(entries)

    def get_history(self, limit: int = 50) -> list:
        entries = self._read()
        return entries[-limit:]

    def clear(self):
        self._write([])


class PresetStore:
    """Manages named LLM setting presets saved to a JSON file."""

    def __init__(self, file_path: str = "./data/llm_presets.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        if self.file_path.exists():
            try:
                return json.loads(self.file_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _write(self, presets: dict):
        self.file_path.write_text(json.dumps(presets, indent=2, default=str), encoding="utf-8")

    def list(self) -> list[dict]:
        presets = self._read()
        return [{"name": k, **v} for k, v in presets.items()]

    def get(self, name: str) -> dict | None:
        return self._read().get(name)

    def save(self, name: str, settings: dict):
        presets = self._read()
        presets[name] = {k: v for k, v in settings.items() if k in ("model", "temperature", "top_p", "top_k")}
        self._write(presets)

    def delete(self, name: str) -> bool:
        presets = self._read()
        if name not in presets:
            return False
        del presets[name]
        self._write(presets)
        return True