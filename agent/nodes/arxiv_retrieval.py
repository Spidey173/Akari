"""
Node 2: arXiv Retrieval

Calls the official arXiv Atom API (export.arxiv.org/api/query) directly via
requests + feedparser rather than the third-party `arxiv` pip package, to
keep the dependency surface minimal and network interactions transparent.

Handles the "zero or many candidates" failure case explicitly (see README "Design Decisions"):
  - Zero results -> status=FAILED with a clear message, no crash.
  - Many results -> capped at MAX_RESULTS, ranked downstream, not silently
    truncated in a way that hides how many existed.
"""

from __future__ import annotations

import time
import urllib.parse

import feedparser
import requests

from agent.state import GraphState, NodeStatus, PaperMetadata

ARXIV_API = "http://export.arxiv.org/api/query"
MAX_RESULTS = 10
NS = {"arxiv": "http://arxiv.org/schemas/atom"}


def _entry_to_metadata(entry) -> PaperMetadata:
    arxiv_id = entry.id.split("/abs/")[-1]
    arxiv_id = arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id.split("/")[-1] else arxiv_id
    pdf_url = next(
        (link.href for link in entry.links if getattr(link, "title", "") == "pdf"),
        entry.id.replace("/abs/", "/pdf/"),
    )
    categories = [t["term"] for t in entry.get("tags", [])] if entry.get("tags") else []
    return PaperMetadata(
        arxiv_id=arxiv_id,
        title=" ".join(entry.title.split()),
        authors=[a.name for a in entry.get("authors", [])],
        abstract=" ".join(entry.summary.split()),
        pdf_url=pdf_url,
        published=entry.get("published", "")[:10],
        categories=categories,
    )


def _query_arxiv(search_query: str, max_results: int, id_list: str = "") -> list:
    params = {
        "search_query": search_query,
        "id_list": id_list,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            return feed.entries
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"arXiv API request failed after retries: {last_err}")


def run(state: GraphState) -> GraphState:
    try:
        if state.intent == "specific_paper" and state.explicit_arxiv_id:
            entries = _query_arxiv("", max_results=1, id_list=state.explicit_arxiv_id)
            if not entries:
                state.fail(
                    "arxiv_retrieval", f"No paper found for arXiv ID {state.explicit_arxiv_id}"
                )
                state.status = NodeStatus.FAILED
                return state
            state.candidates = [_entry_to_metadata(entries[0])]
            state.log("arxiv_retrieval", f"Fetched exact paper {state.explicit_arxiv_id}")

        else:
            search_expr = f"all:{state.search_query}"
            entries = _query_arxiv(search_expr, max_results=MAX_RESULTS)

            if not entries:
                state.fail(
                    "arxiv_retrieval",
                    f"No candidates found for topic '{state.search_query}'. "
                    "Try a broader or differently-worded query.",
                )
                state.status = NodeStatus.FAILED
                return state

            state.candidates = [_entry_to_metadata(e) for e in entries]
            state.log(
                "arxiv_retrieval",
                f"Found {len(state.candidates)} candidate(s) for topic search "
                f"(capped at {MAX_RESULTS}).",
            )

        state.status = NodeStatus.OK

    except Exception as e:  # noqa: BLE001
        state.fail("arxiv_retrieval", str(e))
        state.status = NodeStatus.FAILED

    return state
