#!/usr/bin/env python3
"""
Akari (明かり): Autonomous Research Paper Digest & QA Agent.

Usage:
    python main.py "recent work on KV-cache compression for LLMs"
    python main.py 2401.12345
    python main.py https://arxiv.org/abs/2401.12345
    python main.py --resume 2401.12345     # instant resumption of previously indexed paper
    python main.py --list-sessions         # view all cached papers in local database
    python main.py "topic" --interactive   # interactively select from top candidates

If run with no arguments, prompts interactively.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from agent.graph import run_pipeline
from agent.llm_client import build_llm_client
from agent.nodes import qa
from agent.session import list_sessions, load_session, save_session
from agent.state import Briefing, GraphState
from agent.ui import (
    choose_candidate_interactively,
    console,
    display_cached_sessions,
    print_banner,
    print_briefing_rich,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def briefing_to_dict(state: GraphState) -> dict:
    b = state.briefing
    return {
        "title": b.title,
        "authors": b.authors,
        "arxiv_id": b.arxiv_id,
        "published": b.published,
        "link": b.link,
        "summary": b.summary,
        "problem_statement": b.problem_statement,
        "method_bullets": b.method_bullets,
        "key_results": b.key_results,
        "limitations": b.limitations,
        "followup_questions": b.followup_questions,
        "parse_degraded": state.parse_degraded,
        "num_chunks": len(state.chunks),
        "qa_history": [
            {"question": ex.question, "answer": ex.answer, "grounded": ex.grounded}
            for ex in state.qa_history
        ],
    }


def run_qa_loop(state: GraphState, llm) -> None:
    console.print("[bold cyan]Ask follow-up questions about the paper[/bold cyan] [dim](type 'exit' or 'quit' to stop)[/dim]:\n")
    while True:
        try:
            question = console.input("[bold green]Q:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        exchange = qa.answer_question(state, llm, question)
        if exchange.grounded:
            console.print(f"[bold cyan]A:[/bold cyan] {exchange.answer}\n")
        else:
            console.print(f"[bold cyan]A:[/bold cyan] {exchange.answer} [yellow][low confidence / ungrounded in paper][/yellow]\n")

        # Save session incrementally as QA exchanges occur
        try:
            save_session(state)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Akari (明かり): Autonomous Research Paper Digest & QA Agent")
    parser.add_argument("query", nargs="?", help="Topic, arXiv ID, or arXiv URL")
    parser.add_argument("--provider", default=None, choices=["gemini", "groq", "ollama"], help="LLM provider (defaults to available key)")
    parser.add_argument("--save", default=None, help="Custom path to save the briefing JSON")
    parser.add_argument("--no-qa", action="store_true", help="Skip the interactive QA loop")
    parser.add_argument("--interactive", action="store_true", help="Interactively choose from ranked candidates on topic search")
    parser.add_argument("--resume", default=None, help="Resume QA on a previously indexed paper by arXiv ID")
    parser.add_argument("--list-sessions", action="store_true", help="List all previously cached paper digests")
    args = parser.parse_args()

    print_banner()

    # 1. Handle --list-sessions
    if args.list_sessions:
        sessions = list_sessions()
        display_cached_sessions(sessions)
        return

    # Initialize LLM client
    try:
        llm = build_llm_client(args.provider)
    except Exception as e:
        console.print(f"\n[red]Failed to initialize LLM client ({args.provider}):[/red] {e}")
        sys.exit(1)

    # 2. Handle --resume <arxiv_id>
    if args.resume:
        cached_state = load_session(args.resume)
        if not cached_state:
            console.print(f"[red]Error:[/red] No cached session found for arXiv ID: [bold]{args.resume}[/bold]")
            console.print("Run [cyan]python main.py --list-sessions[/cyan] to view cached papers.")
            sys.exit(1)

        console.print(f"\n[green]✓ Restored session for paper:[/green] [bold white]{cached_state.selected_paper.title}[/bold white]")
        print_briefing_rich(cached_state.briefing, parse_degraded=cached_state.parse_degraded)
        if not args.no_qa:
            run_qa_loop(cached_state, llm)
            save_session(cached_state)
        return

    # 3. Handle query input
    query = args.query
    if not query:
        query = console.input("\n[bold yellow]Enter a research topic, arXiv ID, or arXiv URL:[/bold yellow] ").strip()
        if not query:
            console.print("[red]No query provided. Exiting.[/red]")
            sys.exit(1)

    # Run the stateful graph
    picker = choose_candidate_interactively if args.interactive else None
    state = run_pipeline(query, llm, interactive_candidate_picker=picker)

    if state.status.value == "failed":
        console.print("\n[bold red]Run failed. Errors:[/bold red]")
        for err in state.errors:
            console.print(f"  [red]•[/red] {err}")
        sys.exit(1)

    # Display Rich briefing
    print_briefing_rich(state.briefing, parse_degraded=state.parse_degraded)

    # Save to SQLite session database
    try:
        save_session(state)
    except Exception as e:
        console.print(f"[dim]Note: Failed to cache session in SQLite: {e}[/dim]")

    # Save to export JSON file
    save_path = args.save or f"briefing_{state.selected_paper.arxiv_id.replace('.', '_')}.json"
    with open(save_path, "w") as f:
        json.dump(briefing_to_dict(state), f, indent=2)
    console.print(f"[dim](Briefing saved to {save_path}; session cached for instant resumption)[/dim]\n")

    # Interactive QA loop
    if not args.no_qa:
        run_qa_loop(state, llm)
        with open(save_path, "w") as f:
            json.dump(briefing_to_dict(state), f, indent=2)
        save_session(state)
        console.print(f"\n[green]Final briefing and QA history saved to {save_path}[/green]")


if __name__ == "__main__":
    main()
