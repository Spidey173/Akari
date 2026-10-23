"""
Node 3: Hybrid Semantic + Lexical Ranking (only runs for topic_search intent)

Combines:
1. Lexical Token Overlap (exact title/abstract matches)
2. Dense Semantic Cosine Similarity via local SentenceTransformer ('all-MiniLM-L6-v2')
3. Reciprocal Rank Fusion (RRF) for robust, balanced multi-candidate scoring.

This solves vocabulary mismatch (e.g. searching "sparse attention" finds papers
mentioning "submodular KV eviction" or "linear complexity transformers") without
requiring external LLM API calls.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

from agent.state import GraphState, NodeStatus

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "on",
    "in",
    "and",
    "or",
    "to",
    "with",
    "recent",
    "work",
    "paper",
    "papers",
    "about",
    "using",
    "based",
}

_EMBEDDER: Any = None


def get_embedder() -> Any:
    """Lazily loads local SentenceTransformer to avoid overhead if specific paper ID was given.
    Returns None if offline test mode is active or loading fails."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER if _EMBEDDER is not False else None

    if os.environ.get("AKARI_OFFLINE_TEST") == "1":
        return None

    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _EMBEDDER = False

    return _EMBEDDER if _EMBEDDER is not False else None


def set_embedder(embedder: Any) -> None:
    """Sets embedder instance for testing / offline injection."""
    global _EMBEDDER
    _EMBEDDER = embedder


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _lexical_score(query_tokens: set[str], title: str, abstract: str) -> float:
    title_tokens = _tokenize(title)
    abstract_tokens = _tokenize(abstract)
    title_overlap = len(query_tokens & title_tokens)
    abstract_overlap = len(query_tokens & abstract_tokens)
    return title_overlap * 2.0 + abstract_overlap * 1.0


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def run(state: GraphState) -> GraphState:
    if state.intent == "specific_paper":
        if state.candidates:
            state.selected_paper = state.candidates[0]
            state.log("ranking", "Skipped ranking (specific paper lookup).")
        return state

    if not state.candidates:
        state.fail("ranking", "No candidates to rank.")
        state.status = NodeStatus.FAILED
        return state

    query_tokens = _tokenize(state.search_query)

    # 1. Compute Lexical Scores
    for paper in state.candidates:
        paper.score = _lexical_score(query_tokens, paper.title, paper.abstract)

    # 2. Compute Dense Semantic Similarity if local embedder is available
    embedder = get_embedder()
    if embedder is not None and len(state.candidates) > 1:
        try:
            query_emb = embedder.encode(state.search_query, normalize_embeddings=True)
            paper_texts = [f"{p.title}. {p.abstract[:500]}" for p in state.candidates]
            cand_embs = embedder.encode(paper_texts, normalize_embeddings=True)

            # Reciprocal Rank Fusion (RRF) with k=60
            RRF_K = 60
            lexical_ranked = sorted(state.candidates, key=lambda p: p.score, reverse=True)
            lex_ranks = {p.arxiv_id: rank for rank, p in enumerate(lexical_ranked)}

            dense_sims = [float(_cosine_similarity(query_emb, c_emb)) for c_emb in cand_embs]
            dense_pairs = list(zip(state.candidates, dense_sims, strict=True))
            dense_ranked = sorted(dense_pairs, key=lambda pair: pair[1], reverse=True)
            dense_ranks = {p.arxiv_id: rank for rank, (p, _) in enumerate(dense_ranked)}

            for p in state.candidates:
                lex_r = lex_ranks[p.arxiv_id]
                dense_r = dense_ranks[p.arxiv_id]
                # Combined RRF score
                rrf_score = (1.0 / (RRF_K + lex_r)) + (1.5 / (RRF_K + dense_r))
                p.score = round(rrf_score * 100, 2)

        except Exception as e:
            state.log("ranking", f"Dense embedding ranking skipped due to exception: {e}")

    state.candidates.sort(key=lambda p: p.score, reverse=True)
    state.selected_paper = state.candidates[0]

    state.log(
        "ranking",
        f"Selected '{state.selected_paper.title[:70]}...' "
        f"(score={state.selected_paper.score:.1f}) out of {len(state.candidates)} candidates.",
    )
    state.status = NodeStatus.OK
    return state
