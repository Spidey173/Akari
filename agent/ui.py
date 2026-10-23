"""
Terminal UI formatting and interactive CLI helpers using Rich.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.state import Briefing, PaperMetadata

console = Console()


def print_banner() -> None:
    title_text = Text("Akari (明かり)", style="bold cyan")
    subtitle = Text(
        "Autonomous Research Assistant • Stateful arXiv Digest & QA Agent", style="italic dim"
    )
    banner = Panel(
        Text.assemble(title_text, "  -  ", subtitle),
        border_style="cyan",
        expand=False,
        padding=(0, 3),
    )
    console.print(banner)


def display_candidates_table(candidates: list[PaperMetadata]) -> None:
    table = Table(
        title="[bold cyan]Discovered arXiv Candidates[/bold cyan]",
        title_justify="left",
        border_style="dim",
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#", justify="center", style="bold yellow", width=4)
    table.add_column("arXiv ID", style="green", width=12)
    table.add_column("Title", style="bold white", ratio=4)
    table.add_column("Published", style="dim", width=12)
    table.add_column("Score", justify="right", style="cyan", width=8)

    for idx, paper in enumerate(candidates, 1):
        table.add_row(
            str(idx),
            paper.arxiv_id,
            paper.title,
            paper.published or "N/A",
            f"{paper.score:.1f}",
        )
    console.print(table)


def choose_candidate_interactively(candidates: list[PaperMetadata]) -> PaperMetadata:
    display_candidates_table(candidates)
    while True:
        try:
            choice = console.input(
                f"\n[bold yellow]Select a paper to analyze [1-{len(candidates)}] (default 1):[/bold yellow] "
            ).strip()
            if not choice:
                return candidates[0]
            val = int(choice)
            if 1 <= val <= len(candidates):
                return candidates[val - 1]
            console.print(f"[red]Please enter a number between 1 and {len(candidates)}.[/red]")
        except ValueError:
            console.print(
                f"[red]Invalid input. Enter a number between 1 and {len(candidates)}.[/red]"
            )


def print_briefing_rich(b: Briefing, parse_degraded: bool = False) -> None:
    authors_str = ", ".join(b.authors) if b.authors else "Unknown Authors"
    meta_info = (
        f"[bold white]Title:[/bold white] [bold cyan]{b.title}[/bold cyan]\n"
        f"[bold white]Authors:[/bold white] {authors_str}\n"
        f"[bold white]arXiv ID:[/bold white] [green]{b.arxiv_id}[/green]  •  "
        f"[bold white]Published:[/bold white] {b.published}  •  "
        f"[bold white]Link:[/bold white] [link={b.link}]{b.link}[/link]"
    )

    if parse_degraded:
        meta_info += "\n\n[bold yellow]⚠️ Note: Full-text extraction was degraded (e.g. scanned PDF). Briefing generated primarily from abstract.[/bold yellow]"

    console.print("\n")
    console.print(
        Panel(
            meta_info, title="[bold cyan]Executive Paper Briefing[/bold cyan]", border_style="cyan"
        )
    )

    # Sections
    console.print("\n[bold magenta]1. Why this paper matters[/bold magenta]")
    console.print(Panel(b.summary, border_style="dim"))

    console.print("[bold magenta]2. Problem Statement[/bold magenta]")
    console.print(Panel(b.problem_statement, border_style="dim"))

    console.print("[bold magenta]3. Method & Approach[/bold magenta]")
    for item in b.method_bullets:
        console.print(f"  [cyan]•[/cyan] {item}")

    console.print("\n[bold magenta]4. Key Results & Claims[/bold magenta]")
    for item in b.key_results:
        console.print(f"  [green]•[/green] {item}")

    console.print("\n[bold magenta]5. Limitations & Caveats[/bold magenta]")
    for item in b.limitations:
        console.print(f"  [yellow]•[/yellow] {item}")

    console.print("\n[bold magenta]6. Suggested Follow-up Questions[/bold magenta]")
    for item in b.followup_questions:
        console.print(f"  [blue]?[/blue] {item}")

    console.print("\n" + "─" * 78 + "\n")


def display_cached_sessions(sessions: list[dict]) -> None:
    if not sessions:
        console.print("[yellow]No cached paper sessions found in database.[/yellow]")
        return

    table = Table(
        title="[bold cyan]Akari - Cached Research Sessions[/bold cyan]",
        border_style="dim",
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("arXiv ID", style="bold green", width=14)
    table.add_column("Title", style="white", ratio=4)
    table.add_column("Published", style="dim", width=12)

    for s in sessions:
        table.add_row(s["arxiv_id"], s["title"], s.get("published", "N/A") or "N/A")

    console.print(table)
    console.print("\n[dim]To resume QA on any indexed paper, run:[/dim]")
    console.print("  [cyan]python main.py --resume <arxiv_id>[/cyan]\n")
