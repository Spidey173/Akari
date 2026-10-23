"""
Node 5: Chunk & Embed

Chunking strategy: fixed-size character windows with overlap, aligned to
sentence boundaries where possible. We chunk per-section (not the raw
concatenated text) so each chunk keeps a `section` tag - this lets the QA
node's citations say "this comes from the Method section" and lets us
weight/filter retrieval by section later if needed.

Embeddings: sentence-transformers `all-MiniLM-L6-v2` running locally (free,
no API key, ~80MB download on first run) via ChromaDB's embedding function
interface. This keeps the whole pipeline runnable with zero paid keys even
if the LLM call itself uses a hosted free tier.

Vector DB: Chroma in local persistent mode (durable data directory under
~/.cache/akari/chroma by default), with run-isolated collections to eliminate
concurrency races between parallel runs.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from agent.config import CHROMA_DIR
from agent.embedding import get_chroma_client, get_embedding_function
from agent.state import Chunk, GraphState, NodeStatus

CHUNK_SIZE = 1000  # chars
CHUNK_OVERLAP = 150  # chars


def _sentence_aware_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Greedy chunking that tries to break on sentence boundaries near the
    target size rather than mid-word, falling back to a hard cut if a single
    'sentence' is longer than the chunk size."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            if len(sent) > size:
                # single sentence too long -> hard split
                for i in range(0, len(sent), size - overlap):
                    chunks.append(sent[i : i + size])
                current = ""
            else:
                current = sent

    if current:
        chunks.append(current)

    # add overlap by prepending tail of previous chunk
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail} {c}")
    return overlapped


def run(state: GraphState, chroma_path: Optional[str] = None) -> GraphState:
    if not state.sections:
        state.fail("chunk_embed", "No parsed sections available to chunk.")
        state.status = NodeStatus.FAILED
        return state

    if not state.selected_paper:
        state.fail("chunk_embed", "No paper selected.")
        state.status = NodeStatus.FAILED
        return state

    try:
        chunks: list[Chunk] = []
        chunk_idx = 0
        clean_arxiv_id = state.selected_paper.arxiv_id.replace(".", "_").replace("/", "_")

        for section_name, section_text in state.sections.items():
            if not section_text.strip():
                continue
            pieces = _sentence_aware_chunks(section_text, CHUNK_SIZE, CHUNK_OVERLAP)
            offset = 0
            for piece in pieces:
                start = section_text.find(piece[:50]) if len(piece) >= 50 else offset
                start = max(start, 0)
                chunks.append(
                    Chunk(
                        chunk_id=f"{clean_arxiv_id}_{chunk_idx}",
                        text=piece,
                        section=section_name,
                        char_start=start,
                        char_end=start + len(piece),
                    )
                )
                chunk_idx += 1
                offset = start + len(piece)

        if not chunks:
            state.fail("chunk_embed", "Chunking produced zero chunks.")
            state.status = NodeStatus.FAILED
            return state

        state.chunks = chunks

        client = get_chroma_client(path=chroma_path or CHROMA_DIR)
        embed_fn = get_embedding_function()

        # Generate a run-scoped collection name to guarantee concurrency isolation.
        # This prevents concurrent indexing or QA runs from colliding or deleting active collections.
        run_tag = uuid.uuid4().hex[:8]
        collection_name = f"paper_{clean_arxiv_id}_{run_tag}"

        collection = client.create_collection(
            name=collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"section": c.section} for c in chunks],
        )

        state.collection_name = collection_name
        state.log(
            "chunk_embed",
            f"Created {len(chunks)} chunks across {len(state.sections)} section(s), "
            f"embedded into isolated Chroma collection '{collection_name}'.",
        )
        state.status = NodeStatus.OK

    except Exception as e:  # noqa: BLE001
        state.fail("chunk_embed", str(e))
        state.status = NodeStatus.FAILED

    return state
