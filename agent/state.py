"""
Shared state object that flows through every node in the graph.

Design choice: a single mutable dataclass instance is threaded through all
nodes (rather than each node returning a partial dict that gets merged, as
LangGraph does). For a single-process CLI tool this is simpler to reason
about and debug, at the cost of losing LangGraph's automatic state-merging
and checkpointing. See README "Design Decisions" for the tradeoff discussion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NodeStatus(Enum):
    OK = "ok"
    RETRY = "retry"
    FAILED = "failed"          # unrecoverable for this node, but graph can degrade gracefully
    NEEDS_USER_INPUT = "needs_user_input"


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    published: str
    categories: list[str]
    score: float = 0.0          # relevance score, set during ranking


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section: str
    char_start: int
    char_end: int


@dataclass
class QAExchange:
    question: str
    answer: str
    grounded: bool
    retrieved_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Briefing:
    title: str
    authors: list[str]
    arxiv_id: str
    published: str
    link: str
    summary: str
    problem_statement: str
    method_bullets: list[str]
    key_results: list[str]
    limitations: list[str]
    followup_questions: list[str]


@dataclass
class GraphState:
    # ---- input ----
    raw_query: str = ""

    # ---- query understanding ----
    intent: str = ""                       # "topic_search" | "specific_paper"
    search_query: str = ""                 # cleaned query for arXiv API
    explicit_arxiv_id: Optional[str] = None

    # ---- retrieval ----
    candidates: list[PaperMetadata] = field(default_factory=list)

    # ---- selection ----
    selected_paper: Optional[PaperMetadata] = None

    # ---- fetch & parse ----
    pdf_path: Optional[str] = None
    full_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    parse_degraded: bool = False           # True if we fell back to abstract-only etc.

    # ---- chunk & embed ----
    chunks: list[Chunk] = field(default_factory=list)
    collection_name: Optional[str] = None

    # ---- summarize ----
    briefing: Optional[Briefing] = None

    # ---- QA ----
    qa_history: list[QAExchange] = field(default_factory=list)

    # ---- control / bookkeeping ----
    status: NodeStatus = NodeStatus.OK
    errors: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, node: str, message: str) -> None:
        self.trace.append({"t": time.time(), "node": node, "message": message})

    def fail(self, node: str, message: str) -> None:
        self.errors.append(f"[{node}] {message}")
        self.log(node, f"ERROR: {message}")
