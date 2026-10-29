"""
Embedding and vector client abstractions for Akari.

Provides unified Chroma client access and embedding functions, with native
support for offline/deterministic execution during testing without contacting
external model hubs or Hugging Face.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import warnings
from typing import Any, Optional

from agent.config import CHROMA_DIR, EMBEDDING_MODEL_NAME

warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")

_OVERRIDE_EMBEDDING_FUNCTION: Any = None
_OVERRIDE_CHROMA_CLIENT: Any = None


class MockEmbeddingFunction:
    """Deterministic, token-aware offline embedding function for testing and evaluation.
    Produces semantic vectors based on normalized token feature hashing without network access.
    Conforms to the Chroma EmbeddingFunction protocol."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in input:
            words = re.findall(r"[a-zA-Z0-9]+", text.lower())
            vec = [0.0] * self.dim
            if not words:
                embeddings.append(vec)
                continue
            for w in words:
                idx = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


def get_embedding_function(model_name: Optional[str] = None) -> Any:
    """Returns the configured Chroma embedding function.
    If an override is set (e.g. during unit tests) or AKARI_OFFLINE_TEST=1,
    returns a MockEmbeddingFunction to prevent network downloads."""
    global _OVERRIDE_EMBEDDING_FUNCTION
    if _OVERRIDE_EMBEDDING_FUNCTION is not None:
        return _OVERRIDE_EMBEDDING_FUNCTION

    if os.environ.get("AKARI_OFFLINE_TEST") == "1":
        return MockEmbeddingFunction()

    from chromadb.utils import embedding_functions

    model = model_name or EMBEDDING_MODEL_NAME
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)  # type: ignore[attr-defined]


def set_embedding_function(fn: Any) -> None:
    """Sets a global embedding function override (useful for tests)."""
    global _OVERRIDE_EMBEDDING_FUNCTION
    _OVERRIDE_EMBEDDING_FUNCTION = fn


def get_chroma_client(path: Optional[str] = None, ephemeral: bool = False) -> Any:
    """Returns a ChromaDB client (PersistentClient or EphemeralClient)."""
    global _OVERRIDE_CHROMA_CLIENT
    if _OVERRIDE_CHROMA_CLIENT is not None:
        return _OVERRIDE_CHROMA_CLIENT

    import chromadb

    if ephemeral or os.environ.get("AKARI_EPHEMERAL_CHROMA") == "1":
        return chromadb.EphemeralClient()

    target_dir = path or CHROMA_DIR
    os.makedirs(target_dir, exist_ok=True)
    return chromadb.PersistentClient(path=target_dir)


def set_chroma_client(client: Any) -> None:
    """Sets a global Chroma client override (useful for tests)."""
    global _OVERRIDE_CHROMA_CLIENT
    _OVERRIDE_CHROMA_CLIENT = client
