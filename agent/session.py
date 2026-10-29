"""
Persistent session caching layer for Akari.

Stores processed paper digests and index metadata in a local SQLite database
(~/.cache/akari/sessions.db or custom path), enabling:
1. Instant resumption into interactive QA without re-downloading or re-embedding.
2. Listing previously analyzed papers (`python main.py --list-sessions`).
3. Persistent caching across terminal sessions and reboots.
4. Concurrent-safe operations with SQLite WAL journal mode and busy timeouts.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Optional

from agent.config import DEFAULT_DB_PATH
from agent.state import Briefing, GraphState, NodeStatus, PaperMetadata, QAExchange

logger = logging.getLogger(__name__)


def get_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Opens a SQLite connection with WAL mode and concurrency busy timeouts."""
    db_dir = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                published TEXT,
                pdf_url TEXT,
                collection_name TEXT NOT NULL,
                briefing_json TEXT NOT NULL,
                qa_history_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
    return conn


def save_session(state: GraphState, db_path: str = DEFAULT_DB_PATH) -> None:
    """Persists or updates an active GraphState session.
    Logs warnings on serialization or database errors without crashing the agent."""
    if not state.selected_paper or not state.briefing or not state.collection_name:
        return

    try:
        conn = get_db(db_path)
        now = time.time()
        briefing_dict = {
            "title": state.briefing.title,
            "authors": state.briefing.authors,
            "arxiv_id": state.briefing.arxiv_id,
            "published": state.briefing.published,
            "link": state.briefing.link,
            "summary": state.briefing.summary,
            "problem_statement": state.briefing.problem_statement,
            "method_bullets": state.briefing.method_bullets,
            "key_results": state.briefing.key_results,
            "limitations": state.briefing.limitations,
            "followup_questions": state.briefing.followup_questions,
            "section_citations": getattr(state.briefing, "section_citations", {}),
        }
        qa_list = [
            {
                "question": q.question,
                "answer": q.answer,
                "grounded": q.grounded,
                "retrieved_chunk_ids": q.retrieved_chunk_ids,
                "sources": getattr(q, "sources", []),
                "cited_sections": getattr(q, "cited_sections", []),
            }
            for q in state.qa_history
        ]

        with conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    arxiv_id, title, authors, published, pdf_url,
                    collection_name, briefing_json, qa_history_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    collection_name=excluded.collection_name,
                    briefing_json=excluded.briefing_json,
                    qa_history_json=excluded.qa_history_json,
                    updated_at=excluded.updated_at
                """,
                (
                    state.selected_paper.arxiv_id,
                    state.selected_paper.title,
                    json.dumps(state.selected_paper.authors),
                    state.selected_paper.published,
                    state.selected_paper.pdf_url,
                    state.collection_name,
                    json.dumps(briefing_dict),
                    json.dumps(qa_list),
                    now,
                    now,
                ),
            )
    except Exception as e:
        logger.error(
            "Failed to save session for arXiv ID %s to %s: %s",
            getattr(state.selected_paper, "arxiv_id", "unknown"),
            db_path,
            e,
        )
        raise


def load_session(arxiv_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[GraphState]:
    """Loads a previously indexed paper into a GraphState ready for instant QA."""
    conn = get_db(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE arxiv_id = ?", (arxiv_id.strip(),))
    row = cur.fetchone()
    if not row:
        return None

    authors = json.loads(row["authors"])
    b_data = json.loads(row["briefing_json"])
    qa_data = json.loads(row["qa_history_json"])

    paper = PaperMetadata(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        authors=authors,
        abstract="",
        pdf_url=row["pdf_url"],
        published=row["published"] or "",
        categories=[],
    )

    briefing = Briefing(
        title=b_data["title"],
        authors=b_data["authors"],
        arxiv_id=b_data["arxiv_id"],
        published=b_data["published"],
        link=b_data["link"],
        summary=b_data["summary"],
        problem_statement=b_data["problem_statement"],
        method_bullets=b_data["method_bullets"],
        key_results=b_data["key_results"],
        limitations=b_data["limitations"],
        followup_questions=b_data["followup_questions"],
        section_citations=b_data.get("section_citations", {}),
    )

    qa_history = [
        QAExchange(
            question=item["question"],
            answer=item["answer"],
            grounded=item.get("grounded", True),
            retrieved_chunk_ids=item.get("retrieved_chunk_ids", []),
            sources=item.get("sources", []),
            cited_sections=item.get("cited_sections", []),
        )
        for item in qa_data
    ]

    state = GraphState(
        raw_query=row["arxiv_id"],
        intent="specific_paper",
        explicit_arxiv_id=row["arxiv_id"],
        selected_paper=paper,
        collection_name=row["collection_name"],
        briefing=briefing,
        qa_history=qa_history,
        status=NodeStatus.OK,
    )
    return state


def list_sessions(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Returns metadata for all cached sessions."""
    conn = get_db(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT arxiv_id, title, published, updated_at FROM sessions ORDER BY updated_at DESC"
    )
    rows = cur.fetchall()
    return [
        {
            "arxiv_id": r["arxiv_id"],
            "title": r["title"],
            "published": r["published"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]
