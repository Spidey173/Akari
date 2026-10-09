"""
Node 1: Query Understanding

Decides whether the user gave us a specific paper (arXiv ID or URL) or a
topic to search for. This is done with regex first (cheap, deterministic,
no LLM call needed for the common case) and only falls back to nothing
fancier than string cleanup for topic queries — we deliberately don't spend
an LLM call here since the distinction is a simple pattern match.
"""

from __future__ import annotations

import re

from agent.state import GraphState, NodeStatus

# Matches "2401.12345", "2401.12345v2", with or without an arxiv.org URL wrapper.
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def run(state: GraphState) -> GraphState:
    query = state.raw_query.strip()
    match = ARXIV_ID_RE.search(query)

    if match:
        state.intent = "specific_paper"
        state.explicit_arxiv_id = match.group(1)
        state.log("query_understanding", f"Detected explicit arXiv ID: {state.explicit_arxiv_id}")
    else:
        state.intent = "topic_search"
        state.search_query = query
        state.log("query_understanding", f"Treating as topic search: '{query}'")

    if not query:
        state.fail("query_understanding", "Empty query provided.")
        state.status = NodeStatus.FAILED

    return state
