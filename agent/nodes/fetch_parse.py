"""
Node 4: Layout-Aware Fetch & Parse with Multi-Format Fallback

1. Layout-Aware Two-Column PDF Parsing:
   Uses PyMuPDF's block-level spatial geometry (`get_text("blocks")`) to sort
   text blocks by columns (x0, y0) before stitching. This eliminates the
   notorious two-column horizontal text interleaving bug on IEEE/ACM/arXiv formats.

2. Automated ar5iv HTML Fallback:
   If PDF text yields suspiciously low content or layout errors, Akari attempts
   to fetch the clean HTML version from ar5iv.labs.arxiv.org (which features
   native MathML and clean table structures) before degrading to abstract-only.

3. Graceful Degradation:
   Falls back to abstract-only processing if all parsing vectors fail.

4. Concurrent-Safe Atomic Downloads:
   Downloads write to an isolated temporary file before atomically renaming to
   destination, avoiding race conditions and corrupted reads in parallel runs.
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from typing import Optional

import requests

from agent.config import DOWNLOAD_DIR
from agent.state import GraphState, NodeStatus

logger = logging.getLogger(__name__)

MAX_CHARS = 120_000  # ~30k tokens, comfortably covers a 15-20 page paper
MIN_CHARS_PER_PAGE = 200  # below this average, assume scanned/garbled -> degrade

SECTION_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(abstract|introduction|related work|background|"
    r"method(?:ology)?|approach|experiments?|results?|discussion|"
    r"limitations?|conclusion|references)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _download_pdf(url: str, arxiv_id: str, download_dir: Optional[str] = None) -> str:
    target_dir = download_dir or DOWNLOAD_DIR
    os.makedirs(target_dir, exist_ok=True)
    clean_id = arxiv_id.replace("/", "_")
    path = os.path.join(target_dir, f"{clean_id}.pdf")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    temp_path = os.path.join(target_dir, f"{clean_id}.pdf.tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "akari/1.0"})
            resp.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(resp.content)
            # Atomic rename guarantees concurrent readers never see a partial file
            os.replace(temp_path, path)
            return path
        except Exception as e:
            last_err = e
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            logger.warning("Download attempt %d for %s failed: %s", attempt + 1, url, e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download PDF from {url} after retries: {last_err}")


def _extract_text_layout_aware(pdf_path: str) -> tuple[str, int]:
    """Layout-aware extraction: sorts blocks by column geometry to avoid two-column interleaving."""
    import pymupdf as fitz

    doc = fitz.open(pdf_path)
    all_pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Blocks: (x0, y0, x1, y1, "text", block_no, block_type)
        blocks = page.get_text("blocks")
        # Filter text blocks only (block_type == 0)
        text_blocks = [b for b in blocks if b[6] == 0]

        # Determine if page is likely 2-column by checking horizontal distribution
        page_width = page.rect.width
        mid_x = page_width / 2.0

        # Sort blocks: column 1 (left side) top-to-bottom, then column 2 (right side) top-to-bottom
        def block_sort_key(b, mid=mid_x):
            is_right_col = 1 if b[0] >= mid - 20 else 0
            return (is_right_col, b[1])  # (col, y0)

        sorted_blocks = sorted(text_blocks, key=block_sort_key)
        page_content = "\n\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())
        all_pages_text.append(page_content)

    num_pages = len(doc)
    doc.close()
    return "\n\n".join(all_pages_text), num_pages


def _fetch_ar5iv_html(arxiv_id: str) -> str | None:
    """Attempts to fetch clean HTML from ar5iv if PDF extraction fails."""
    clean_id = arxiv_id.replace("v", "").split("/abs/")[-1]
    url = f"https://ar5iv.labs.arxiv.org/html/{clean_id}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "akari/1.0"})
        if resp.status_code == 200 and len(resp.text) > 2000:
            # Strip HTML tags into clean text
            clean_text = re.sub(r"<[^>]+>", " ", resp.text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text
    except Exception as e:
        logger.debug("ar5iv HTML fetch failed for %s: %s", arxiv_id, e)
    return None


def _split_sections(text: str) -> dict[str, str]:
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return {"body": text}

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            if name not in sections or len(content) > len(sections[name]):
                sections[name] = content
    return sections or {"body": text}


def run(state: GraphState, download_dir: Optional[str] = None) -> GraphState:
    paper = state.selected_paper
    if paper is None:
        state.fail("fetch_parse", "No selected paper to fetch.")
        state.status = NodeStatus.FAILED
        return state

    try:
        pdf_path = _download_pdf(paper.pdf_url, paper.arxiv_id, download_dir=download_dir)
        state.pdf_path = pdf_path
        state.log("fetch_parse", f"Downloaded PDF to {pdf_path}")

        text, num_pages = _extract_text_layout_aware(pdf_path)
        chars_per_page = len(text) / max(num_pages, 1)

        if chars_per_page < MIN_CHARS_PER_PAGE:
            state.log(
                "fetch_parse",
                f"PDF yielded low text ({chars_per_page:.0f} chars/page). Trying ar5iv HTML fallback...",
            )
            html_text = _fetch_ar5iv_html(paper.arxiv_id)
            if html_text and len(html_text) > 2000:
                text = html_text
                state.log("fetch_parse", "Recovered full text via ar5iv HTML extraction.")
                state.parse_degraded = False
            else:
                state.log("fetch_parse", "ar5iv unavailable; falling back to abstract-only mode.")
                state.full_text = paper.abstract
                state.sections = {"abstract": paper.abstract}
                state.parse_degraded = True
                state.status = NodeStatus.OK
                return state

        if len(text) > MAX_CHARS:
            state.log(
                "fetch_parse",
                f"Extracted text ({len(text)} chars) exceeds cap; truncating to {MAX_CHARS}.",
            )
            text = text[:MAX_CHARS]

        state.full_text = text
        state.sections = _split_sections(text)
        state.parse_degraded = False
        state.log(
            "fetch_parse",
            f"Extracted {len(text)} chars across {num_pages} pages (layout-aware), "
            f"{len(state.sections)} section(s) detected.",
        )
        state.status = NodeStatus.OK

    except Exception as e:
        state.log("fetch_parse", f"Fetch/parse failed ({e}); checking ar5iv HTML fallback...")
        html_text = _fetch_ar5iv_html(paper.arxiv_id) if paper else None
        if html_text and len(html_text) > 2000:
            state.full_text = html_text
            state.sections = _split_sections(html_text)
            state.parse_degraded = False
            state.status = NodeStatus.OK
        else:
            state.full_text = paper.abstract
            state.sections = {"abstract": paper.abstract}
            state.parse_degraded = True
            state.status = NodeStatus.OK

    return state
