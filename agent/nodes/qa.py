"""
Node 7: QA Loop

Retrieves top-k relevant chunks from the Chroma collection for a user
question, then asks the LLM to answer strictly grounded in those chunks.

Multi-layer grounding strategy for hallucination mitigation:
  1. Retrieval-constrained context: the LLM only ever sees the retrieved
     chunks, never the full paper - it physically cannot answer from
     content outside what was retrieved.
  2. Explicit instruction + required refusal phrase: the system prompt
     requires the model to say a fixed phrase ("NOT_IN_PAPER") if the
     retrieved chunks don't contain the answer, which we detect and surface
     to the user as "I couldn't find this in the paper" rather than passing
     through a hedge the model might still hallucinate around.
  3. We track `grounded=False` whenever the refusal phrase fires or when
     retrieval similarity is very low, so the CLI can visibly flag
     low-confidence answers instead of presenting everything with equal
     confidence.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.config import CHROMA_DIR
from agent.embedding import get_chroma_client, get_embedding_function
from agent.llm_client import LLMClient
from agent.state import GraphState, QAExchange

TOP_K = 5
REFUSAL_MARKER = "NOT_IN_PAPER"

SYSTEM_PROMPT = f"""You are answering questions about a specific academic paper using ONLY the
retrieved excerpts provided below. You must not use outside knowledge or
general knowledge about the topic beyond what's in the excerpts.

If the excerpts do not contain enough information to answer the question,
respond with exactly: {REFUSAL_MARKER}
Do not guess, speculate, or fill gaps with general knowledge in that case.

When you can answer, be concise and cite which part of the paper you're
drawing from in plain language (e.g. "In the Method section...")."""

PROMPT_TEMPLATE = """Retrieved excerpts from the paper:
{context}

Conversation so far:
{history}

Question: {question}

Answer using only the excerpts above."""


def _format_context(chunks_with_meta: list[tuple[str, str, str]]) -> str:
    # (chunk_id, section, text)
    parts = []
    for chunk_id, section, text in chunks_with_meta:
        parts.append(f"[{chunk_id} | section: {section}]\n{text}")
    return "\n\n".join(parts)


def _format_history(history: list[QAExchange], max_turns: int = 3) -> str:
    if not history:
        return "(none yet)"
    recent = history[-max_turns:]
    return "\n".join(f"Q: {ex.question}\nA: {ex.answer}" for ex in recent)


def answer_question(
    state: GraphState,
    llm: LLMClient,
    question: str,
    chroma_path: Optional[str] = None,
    collection: Optional[Any] = None,
) -> QAExchange:
    """Runs one QA turn. Returns the exchange and appends it to state.qa_history."""
    if collection is None:
        if not state.collection_name:
            exchange = QAExchange(
                question=question,
                answer="I couldn't retrieve any relevant content because no indexed paper collection is active.",
                grounded=False,
            )
            state.qa_history.append(exchange)
            return exchange
        client = get_chroma_client(path=chroma_path or CHROMA_DIR)
        embed_fn = get_embedding_function()
        collection = client.get_collection(name=state.collection_name, embedding_function=embed_fn)

    results = collection.query(query_texts=[question], n_results=TOP_K)

    ids = results["ids"][0] if results["ids"] else []
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    distances = results.get("distances", [[]])[0]

    if not docs:
        exchange = QAExchange(
            question=question,
            answer="I couldn't retrieve any relevant content for this question.",
            grounded=False,
        )
        state.qa_history.append(exchange)
        return exchange

    chunks_with_meta = [(ids[i], metas[i].get("section", "?"), docs[i]) for i in range(len(docs))]
    context = _format_context(chunks_with_meta)
    history_str = _format_history(state.qa_history)

    prompt = PROMPT_TEMPLATE.format(context=context, history=history_str, question=question)
    raw_answer = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1024)

    grounded = REFUSAL_MARKER not in raw_answer
    if not grounded:
        answer_text = (
            "I couldn't find an answer to this in the retrieved parts of the paper. "
            "It may not be covered, or may be in a section that wasn't parsed."
        )
    else:
        answer_text = raw_answer

    # Low-similarity heuristic: with cosine distance (0.0=identical, 1.0=orthogonal),
    # matches above 0.85 represent weak semantic overlap.
    if distances and distances[0] > 0.85:
        grounded = False

    exchange = QAExchange(
        question=question,
        answer=answer_text,
        grounded=grounded,
        retrieved_chunk_ids=ids,
    )
    state.qa_history.append(exchange)
    return exchange
