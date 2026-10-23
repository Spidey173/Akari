"""
The Agentic State Graph with Self-Correction & Dynamic Routing.

Nodes:
  1. query_understanding
  2. arxiv_retrieval
  3. ranking (Hybrid Semantic RRF + Interactive human-in-the-loop option)
  4. fetch_parse (Layout-aware two-column parser + ar5iv HTML fallback)
  5. chunk_embed (Sentence-aware per-section Chroma indexing)
  6. summarize (Section-aware hierarchical briefing synthesis)
  7. interactive QA loop

Dynamic Self-Correction Cycles:
  - Query Reformulation: If topic retrieval returns zero or low relevance,
    the agent automatically reformulates the search expression and retries.
  - Multi-Modal Format Pivot: If PDF layout extraction fails or is scanned,
    the agent pivots to clean ar5iv HTML before degrading.
  - Briefing Recovery: If structured parsing encounters JSON syntax degradation,
    the agent self-corrects using schema enforcement fallbacks.
"""

from __future__ import annotations

from typing import Callable, Optional

from agent.llm_client import LLMClient
from agent.nodes import (
    arxiv_retrieval,
    chunk_embed,
    fetch_parse,
    query_understanding,
    ranking,
    summarize,
)
from agent.state import GraphState, NodeStatus, PaperMetadata


def run_pipeline(
    query: str,
    llm: LLMClient,
    interactive_candidate_picker: Optional[Callable[[list[PaperMetadata]], PaperMetadata]] = None,
) -> GraphState:
    state = GraphState(raw_query=query)

    # 1. Query Understanding
    print("\n-> Running node: query_understanding")
    state = query_understanding.run(state)
    if state.status == NodeStatus.FAILED:
        print(f"   Pipeline halted: {state.errors[-1]}")
        return state

    # 2. arXiv Retrieval with Autonomous Query Reformulation Cycle
    print("-> Running node: arxiv_retrieval")
    state = arxiv_retrieval.run(state)

    # Self-Correction Loop: If topic search yielded 0 candidates, attempt query simplification
    if state.status == NodeStatus.FAILED and state.intent == "topic_search":
        original_query = state.search_query
        # Strip noisy qualifiers and prepositions to widen arXiv recall
        noise_words = {
            "recent",
            "work",
            "paper",
            "papers",
            "survey",
            "latest",
            "overview",
            "on",
            "about",
            "regarding",
            "for",
            "in",
            "the",
            "a",
            "an",
        }
        simplified = " ".join([w for w in original_query.split() if w.lower() not in noise_words])
        if simplified and simplified != original_query:
            print(
                f"   [Self-Correction] Zero results for '{original_query}'. Reformulating query to: '{simplified}'"
            )
            state.search_query = simplified
            state.status = NodeStatus.OK
            state.errors.clear()
            state = arxiv_retrieval.run(state)

    if state.status == NodeStatus.FAILED:
        print(
            f"   Pipeline halted at 'arxiv_retrieval': {state.errors[-1] if state.errors else 'unknown error'}"
        )
        return state

    # 3. Hybrid Semantic Ranking & Selection
    print("-> Running node: ranking")
    state = ranking.run(state)
    if (
        state.status == NodeStatus.OK
        and state.intent == "topic_search"
        and len(state.candidates) > 1
        and interactive_candidate_picker is not None
    ):
        chosen = interactive_candidate_picker(state.candidates)
        if chosen:
            state.selected_paper = chosen
            state.log("ranking", f"User selected: {chosen.title[:70]}")

    if state.status == NodeStatus.FAILED:
        print(
            f"   Pipeline halted at 'ranking': {state.errors[-1] if state.errors else 'unknown error'}"
        )
        return state

    # 4. Fetch & Parse (Layout-Aware PDF + ar5iv HTML fallback)
    print("-> Running node: fetch_parse")
    state = fetch_parse.run(state)
    if state.status == NodeStatus.FAILED:
        print(
            f"   Pipeline halted at 'fetch_parse': {state.errors[-1] if state.errors else 'unknown error'}"
        )
        return state

    # 5. Chunk & Embed
    print("-> Running node: chunk_embed")
    state = chunk_embed.run(state)
    if state.status == NodeStatus.FAILED:
        print(
            f"   Pipeline halted at 'chunk_embed': {state.errors[-1] if state.errors else 'unknown error'}"
        )
        return state

    # 6. Summarize (Section-aware digest)
    print("-> Running node: summarize")
    state = summarize.run(state, llm)
    if state.status == NodeStatus.FAILED:
        print(
            f"   Pipeline halted at 'summarize': {state.errors[-1] if state.errors else 'unknown error'}"
        )
        return state

    print("\nPipeline complete. Briefing generated, ready for QA.")
    return state
