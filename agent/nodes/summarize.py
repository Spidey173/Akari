"""
Node 6: Section-Aware Structured Executive Briefing

Solves the LLM "Lost in the Middle" and context truncation degradation:
Instead of blindly truncating or stuffing 40,000 unformatted characters into
one monolithic prompt, this node extracts prioritized section windows:
- Abstract + Problem statement
- Method / Architecture core (from sections['method'] / sections['approach'])
- Empirical Results & Metrics (from sections['results'] / sections['experiments'])
- Discussion & Limitations (from sections['limitations'] / sections['conclusion'])

This ensures smaller models (Ollama 8B, Gemini Flash, Groq Mixtral) receive
the critical technical nuggets from each part of the paper without hallucinating
or dropping the methodology.
"""

from __future__ import annotations

import json
import re

from agent.llm_client import LLMClient
from agent.state import Briefing, GraphState, NodeStatus

SYSTEM_PROMPT = (
    "You are a careful research analyst producing an executive briefing on an "
    "academic paper. You are precise, you do not invent results the paper does "
    "not report, and you always include genuine limitations even if the paper "
    "downplays them. To prevent confident but unsupported claims, every bullet point "
    "under method_bullets, key_results, and limitations MUST cite where it came from in the paper "
    "by appending '[Section: <Section Name>]' at the end of the bullet. "
    "Respond with ONLY valid JSON, no markdown fences, no preamble."
)

PROMPT_TEMPLATE = """Paper metadata:
Title: {title}
Authors: {authors}
arXiv ID: {arxiv_id}
Published: {published}
Abstract: {abstract}

Extracted Section Digests:
---
{section_content}
---

{degraded_note}

Produce a JSON object with exactly these keys:
{{
  "summary": "<one paragraph, plain English, explaining why this paper matters>",
  "problem_statement": "<1-3 sentences: what problem is being solved [Section: Introduction/Abstract]>",
  "method_bullets": ["<bullet point on approach [Section: Method]>", "..."],
  "key_results": ["<bullet point on a specific result or claim [Section: Results]>", "..."],
  "limitations": ["<an actual limitation, infer from scope/method [Section: Limitations/Discussion]>", "..."],
  "followup_questions": ["<a question a curious reader might ask next>", "..."]
}}

Rules:
- Grounding & Provenance: Every bullet in method_bullets, key_results, and limitations MUST end with a source citation in brackets: [Section: <Section Name>], indicating which part of the extracted section digests supports the claim. Do not emit confident assertions without a source citation.
- limitations must have at least 2 entries. Never return an empty list for limitations.
- method_bullets and key_results should have 3-6 entries each where the content supports it.
- followup_questions should have 3-5 entries.
- If parsing was degraded, base the briefing on the abstract and metadata, noting [Section: Abstract] as the citation.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _format_section_digest(sections: dict[str, str], max_per_section: int = 3500) -> str:
    """Selectively packages key sections to prevent 'lost-in-the-middle' prompt degradation."""
    if not sections or list(sections.keys()) == ["abstract"]:
        return "Abstract:\n" + sections.get("abstract", "")

    # Priority sections for an executive briefing
    priority_order = [
        ("method", ["method", "methodology", "approach", "architecture"]),
        ("results", ["results", "experiments", "evaluation", "empirical evaluation"]),
        ("discussion", ["limitations", "discussion", "conclusion"]),
        ("intro", ["introduction", "background"]),
    ]

    parts = []
    used_keys = set()
    covered_categories = set()

    for category, aliases in priority_order:
        if category in covered_categories:
            continue
        for alias in aliases:
            for sec_name, sec_text in sections.items():
                if sec_name not in used_keys and (alias in sec_name.lower()):
                    used_keys.add(sec_name)
                    snippet = sec_text[:max_per_section].strip()
                    parts.append(f"[{sec_name.upper()}]\n{snippet}")
                    covered_categories.add(category)
                    break
            if category in covered_categories:
                break

    # If priority matching missed other sections, append remaining up to budget
    if not parts:
        for name, text in list(sections.items())[:3]:
            parts.append(f"[{name.upper()}]\n{text[:max_per_section]}")

    return "\n\n".join(parts)


def _ensure_citations(bullets: list[str], default_section: str, parse_degraded: bool) -> list[str]:
    """Ensures each bullet point carries an explicit section citation tag."""
    fallback_tag = "[Section: Abstract]" if parse_degraded else f"[Section: {default_section}]"
    normalized: list[str] = []
    for b in bullets:
        b_str = str(b).strip()
        if not b_str:
            continue
        if not re.search(r"\[Section:\s*[^\]]+\]", b_str, re.IGNORECASE):
            b_str = f"{b_str} {fallback_tag}"
        normalized.append(b_str)
    return normalized


def _build_section_citations(
    method_bullets: list[str],
    key_results: list[str],
    limitations: list[str],
) -> dict[str, list[str]]:
    """Maps normalized section names to the bullet points supported by each section."""
    citations: dict[str, list[str]] = {}
    for bullets in [method_bullets, key_results, limitations]:
        for b in bullets:
            match = re.search(r"\[Section:\s*([^\]]+)\]", b, re.IGNORECASE)
            sec = match.group(1).strip().lower() if match else "unknown"
            citations.setdefault(sec, []).append(b)
    return citations


def run(state: GraphState, llm: LLMClient) -> GraphState:
    paper = state.selected_paper
    if paper is None:
        state.fail("summarize", "No selected paper to summarize.")
        state.status = NodeStatus.FAILED
        return state

    section_content = _format_section_digest(state.sections)
    degraded_note = (
        "NOTE: Full-text parsing was degraded; content above is abstract-only."
        if state.parse_degraded
        else ""
    )

    prompt = PROMPT_TEMPLATE.format(
        title=paper.title,
        authors=", ".join(paper.authors),
        arxiv_id=paper.arxiv_id,
        published=paper.published,
        abstract=paper.abstract,
        section_content=section_content,
        degraded_note=degraded_note,
    )

    try:
        raw = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=2048)
        data = _extract_json(raw)

        raw_method = data.get("method_bullets") or []
        raw_results = data.get("key_results") or []
        raw_limitations = data.get("limitations") or []

        if not raw_limitations:
            raw_limitations = [
                "The model did not surface explicit limitations; treat this briefing as a starting point, not a substitute for reading the paper."
            ]

        # Enforce provenance citations on every bullet to prevent confident but unsupported claims
        method_bullets = _ensure_citations(
            raw_method, "Method", parse_degraded=state.parse_degraded
        )
        key_results = _ensure_citations(raw_results, "Results", parse_degraded=state.parse_degraded)
        limitations = _ensure_citations(
            raw_limitations, "Limitations", parse_degraded=state.parse_degraded
        )

        section_citations = _build_section_citations(method_bullets, key_results, limitations)

        state.briefing = Briefing(
            title=paper.title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            published=paper.published,
            link=f"https://arxiv.org/abs/{paper.arxiv_id}",
            summary=data.get("summary", "").strip(),
            problem_statement=data.get("problem_statement", "").strip(),
            method_bullets=method_bullets,
            key_results=key_results,
            limitations=limitations,
            followup_questions=data.get("followup_questions", []),
            section_citations=section_citations,
        )
        state.log("summarize", "Generated structured section-aware briefing with source citations.")
        state.status = NodeStatus.OK

    except Exception as e:
        state.fail("summarize", f"Briefing generation/parsing failed: {e}")
        state.status = NodeStatus.FAILED

    return state
