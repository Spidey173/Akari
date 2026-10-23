"""
Centralized configuration for Akari.

Provides durable, configurable filesystem paths and default settings.
All paths default to user cache directories (~/.cache/akari) rather than /tmp,
ensuring persistence across reboots and preventing accidental cleanup.
Overrideable via environment variables.
"""

from __future__ import annotations

import os

# Base directory for all persistent data (vector index, cache, sessions)
AKARI_DATA_DIR = os.environ.get(
    "AKARI_DATA_DIR",
    os.path.expanduser("~/.cache/akari"),
)

# ChromaDB persistence directory
CHROMA_DIR = os.environ.get(
    "AKARI_CHROMA_DIR",
    os.path.join(AKARI_DATA_DIR, "chroma"),
)

# Downloaded PDFs directory
DOWNLOAD_DIR = os.environ.get(
    "AKARI_PDF_DIR",
    os.path.join(AKARI_DATA_DIR, "pdfs"),
)

# SQLite session database path
DEFAULT_DB_PATH = os.environ.get(
    "AKARI_DB_PATH",
    os.path.join(AKARI_DATA_DIR, "sessions.db"),
)

# Embedding model identifier
EMBEDDING_MODEL_NAME = os.environ.get(
    "AKARI_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)


# Ensure directories exist
def ensure_directories() -> None:
    os.makedirs(AKARI_DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
