# Akari (明かり) — Autonomous arXiv Paper Digest & QA Agent

[![CI](https://github.com/Spidey173/Akari/actions/workflows/ci.yml/badge.svg)](https://github.com/Spidey173/Akari/actions/workflows/ci.yml)

**Akari** is an autonomous research assistant that takes a research topic or a specific arXiv paper ID, retrieves
and parses the paper, produces a structured executive briefing, and answers
follow-up questions grounded in the paper's content via RAG.

Built as an explicit stateful graph (not a single monolithic prompt) — see
[Architecture](#architecture) below.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Gemini API key: https://aistudio.google.com/apikey
cp .env.example .env
# edit .env and paste your key into GEMINI_API_KEY=

# 3. Run it
python main.py "recent work on KV-cache compression for LLMs"
# or interactive paper selection on topics:
python main.py "sparse attention mechanisms" --interactive
# or direct paper analysis:
python main.py 2401.12345
# or instant resumption on a previously analyzed paper:
python main.py --resume 2401.12345
# or list all cached research sessions:
python main.py --list-sessions
```

The first run will download a small local embedding model
(`all-MiniLM-L6-v2`, ~80MB, via `sentence-transformers`) — this happens once
and is cached locally afterward. No embedding API key is needed; embeddings
run entirely on your machine.

**No paid API keys are required.** The default LLM backend is Gemini's free
tier. Two alternative free/local backends are wired in and swappable via
`--provider`:

```bash
python main.py "your topic" --provider groq     # needs GROQ_API_KEY (free tier)
python main.py "your topic" --provider ollama   # fully local, needs `ollama serve` running
```

---

## Architecture

### State Graph

```
                    ┌─────────────────────────┐
  user query   ───▶ │ 1. Query Understanding  │ (regex router: arXiv ID/URL vs topic)
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  2. arXiv Retrieval   │ (official Atom API via feedparser)
                     └───────────┬───────────┘
                                 │
                       [0 results on topic?]
                                 │
                       yes ──────┴──────▶ [Query Reformulation Loop]
                                          (strips noise qualifiers & retries)
                                 │ no
                                 ▼
                     ┌───────────────────────┐
                     │ 3. Candidate Ranking  │ (Hybrid: Lexical Overlap + Dense MiniLM)
                     │                       │ ➔ Reciprocal Rank Fusion (RRF, k=60)
                     │                       │ ➔ Optional Rich interactive picker
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   4. Fetch & Parse    │ (PyMuPDF column geometry sorting)
                     │                       │ ➔ Pivot to ar5iv HTML on layout failure
                     │                       │ ➔ Graceful degradation to abstract
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   5. Chunk & Embed    │ (Sentence-boundary per-section chunking)
                     │                       │ ➔ Local ChromaDB + all-MiniLM-L6-v2
                     │                       │ ➔ Run-isolated collections
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │     6. Summarize      │ (Structured section-aware extraction)
                     │                       │ ➔ Typed dataclasses & JSON validation
                     └───────────┬───────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     ┌───────────────────────┐       ┌───────────────────────┐
     │  SQLite Session Cache │       │   7. Grounded QA Loop │ ◀─── user questions
     │ (~/.cache/akari/      │       │ (Cosine guardrail +   │
     │   sessions.db, WAL)   │       │  NOT_IN_PAPER refusal)│ ────▶ answers + section
     │ Instant resumption via│       └───────────────────────┘       citations
     │ `python main.py --res`│
     └───────────────────────┘
```

### Shared State

A single `GraphState` dataclass (`agent/state.py`) is threaded through every
node. It holds, among other things:

- `intent`, `explicit_arxiv_id`, `search_query` — from query understanding
- `candidates: list[PaperMetadata]`, `selected_paper` — from retrieval/ranking
- `full_text`, `sections: dict[str, str]`, `parse_degraded: bool` — from parsing
- `chunks: list[Chunk]`, `collection_name` — from chunk/embed (the actual
  vectors live in Chroma's on-disk store, keyed by `collection_name`; the
  state only holds a reference to that collection, not the vectors themselves)
- `briefing: Briefing` — the final structured output
- `qa_history: list[QAExchange]` — every question/answer/grounded-flag from
  the QA loop, persisted into the saved JSON

Each node reads what it needs from state, does its work, and writes results
back onto the same object before handing it to the next node. Failures set
`state.status = NodeStatus.FAILED` and populate `state.errors`, allowing the
graph runner (`agent/graph.py`) to stop cleanly and report exactly which node
failed and why, rather than crashing with a raw traceback.

### Persistence & Concurrency Isolation

State is **in-memory during a single run** (the `GraphState` object lives for
the process's lifetime), while index vectors, downloaded PDFs, and structured
session data are persisted in durable, configurable directories (defaulting to
`~/.cache/akari` rather than ephemeral `/tmp`):

- **Durable Vector Store Persistence**: Vector chunks are persisted in a local Chroma
  store (`~/.cache/akari/chroma`), surviving process termination and system reboots.
- **Run-Isolated Collections**: Each indexing run generates a run-scoped collection ID
  (`paper_<arxiv_id>_<uuid>`), ensuring concurrent pipelines never delete or corrupt
  each other's collections during parallel execution.
- **Atomic File Downloads**: PDF downloads write to an isolated temporary file before
  atomically renaming (`os.replace`) into `~/.cache/akari/pdfs`, preventing race
  conditions and corrupted reads in concurrent runs.
- **SQLite Session Cache (WAL Mode)**: Paper briefings and QA history are stored in
  `~/.cache/akari/sessions.db` with SQLite Write-Ahead Logging (`WAL`) and busy timeouts
  enabled, allowing simultaneous readers without database lock contention.
- **Configurable Paths**: All storage locations can be overridden via environment
  variables (`AKARI_DATA_DIR`, `AKARI_CHROMA_DIR`, `AKARI_PDF_DIR`, `AKARI_DB_PATH`).
- **JSON Export**: An exportable JSON file (`briefing_<id>.json`) is generated
  for programmatic consumption or integration into downstream workflows.

---

## Design Decisions & Tradeoffs

**Transparent Python Dataclasses over Heavy Frameworks.** Rather than adding heavy
framework dependencies (such as LangGraph, CrewAI, or Pydantic) for what is
fundamentally a structured directed pipeline, Akari uses a transparent Python
dataclass runner with explicit node contracts:
- *Why:* Keeps every state transition, failure point, and latency profile readable
  and testable in pure Python without framework abstraction overhead.
- *Tradeoff:* For cyclic exploration with complex multi-turn agent branching, an
  orchestration engine like LangGraph would provide richer checkpointing. We
  deliberately drew the line at deterministic self-correction hooks (query
  reformulation and format fallbacks) to maintain zero unnecessary runtime weight.

**Hybrid Semantic + Lexical Ranking (RRF).** arXiv's API returns results ordered
by relevance, but queries often suffer from vocabulary mismatch (e.g. searching
"sparse attention" when a paper discusses "linear transformers" or "submodular
KV pruning").
- *Implementation:* We combine keyword overlap with dense cosine similarity from
  a local bi-encoder (`all-MiniLM-L6-v2`) using Reciprocal Rank Fusion ($k=60$).
- *Tradeoff:* A cross-encoder or LLM-as-a-judge reranker would provide slightly
  higher precision on nuanced abstracts, but would add either notable CPU inference
  latency or extra API calls on free tiers. RRF strikes a practical sweet spot.

**Column-Aware Extraction & Multi-Format Fallback.** Naive text extractors
frequently read horizontally across two-column scientific PDFs, interleaving
paragraphs into broken text:
- *Implementation:* PyMuPDF block geometry (`page.get_text("blocks")`) is
  partitioned by horizontal midpoint coordinates `(is_right_col, y0)` to preserve
  columnar reading order. If PDF text yield is suspiciously low (scanned or
  corrupted PDFs), the pipeline attempts an automated fetch against **ar5iv clean
  HTML** before degrading gracefully to abstract-only mode.
- *Tradeoff:* Extremely complex multi-column layouts with floating figure captions
  can still occasionally interleave captions into body text. Dedicated vision-based
  document parsers (e.g. Docling or MinerU) provide higher visual fidelity at the
  cost of substantial model size and GPU requirements.

**Multi-Layer Grounding Strategy for QA.** Hallucination mitigation is handled
at three distinct levels in `agent/nodes/qa.py`:
1. *Retrieval-bounded context*: The LLM is provided only the top-$k$ retrieved
   chunks, with strict prompt constraints against ungrounded parametric extrapolation.
2. *Programmatic refusal marker (`NOT_IN_PAPER`)*: The model is instructed to emit
   a specific refusal token if evidence is insufficient. This is trapped in code to
   deliver a transparent disclaimer rather than conversational hedging.
3. *Cosine distance guardrail*: Chroma cosine distances are evaluated; if the best
   match has weak semantic similarity, the CLI tags the output as low confidence.

**Local Embeddings (`all-MiniLM-L6-v2`).** Local sentence-transformers run
embeddings entirely on-device:
- *Why:* Zero embedding API egress costs, no external rate limits, and reproducible
  vector lookups regardless of cloud API status.

**Structured Section Digests over Monolithic Context Stuffing.**
Rather than dumping 40,000 raw unformatted characters into a single LLM prompt,
the summarization node organizes prioritized digests (Methodology Core,
Empirical Results, Limitations, and Discussion).
- *Why:* Mitigates the "lost-in-the-middle" effect on local or smaller models
  (e.g., Llama 3 8B, Gemini Flash), ensuring methodology and caveats aren't
  overshadowed by introduction text.

### Engineering Roadmap

1. **Cross-Encoder Reranking**: Optional cross-encoder stage for top-5 candidates
   when topic search scores are closely tied.
2. **Deep Layout Parsing (OCR/Vision)**: Integration of lightweight vision-based
   parsers (e.g. Docling) for complex tabular benchmark extraction.
3. **Multi-Paper Cross-Synthesis**: Cross-collection querying to synthesize findings
   and comparative tables across multiple related papers simultaneously.

*(Note: Fine-grained section citations for both briefings and QA, alongside the quantitative citation accuracy and answer quality evaluation benchmark, are fully implemented.)*

---

## Known Limitations & Operational Considerations

- **Section Splitting Heuristic**: Section boundaries rely on regex heading
  matching. Papers with unconventional styling or numbered sub-headings may group
  subsections under `"body"`, though text remains fully indexed.
- **Single-Paper Focus**: ChromaDB collections and QA sessions currently index one
  paper at a time. Cross-paper synthesis across multiple documents requires
  querying each session independently.
- **Hosted LLM Quotas**: When using free-tier providers (Gemini/Groq), modest
  per-minute and per-day rate limits apply. Exponential backoff handles transient
  429s, but sustained heavy queries benefit from running local inference via `--provider ollama`.

---

## Example Run

```
$ python main.py "recent work on KV-cache compression for LLMs"

-> Running node: query_understanding
-> Running node: arxiv_retrieval
-> Running node: ranking
-> Running node: fetch_parse
-> Running node: chunk_embed
-> Running node: summarize

Pipeline complete. Briefing generated, ready for QA.

==============================================================================
EXECUTIVE BRIEFING: H_2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
==============================================================================
Authors:    Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen
arXiv ID:   2306.14048
Published:  2023-06-24
Link:       https://arxiv.org/abs/2306.14048

--- Why this paper matters ---
Large Language Models (LLMs) suffer from severe memory bottlenecks during generation because the key-value (KV) cache grows linearly with sequence length and batch size. This paper reveals that a small fraction of tokens ("Heavy Hitters") contributes most of the value when computing attention scores. By introducing Heavy Hitter Oracle (H2O), a dynamic eviction policy retaining a balance of recent and heavy-hitter tokens, the authors cut KV cache memory by up to 80% while preserving generation accuracy and accelerating inference throughput up to 29x over standard systems.

--- Problem Statement ---
Deploying LLMs for long-sequence tasks (dialogue, long-document reasoning) is bottlenecked by the transient KV cache stored in GPU memory. Existing memory-saving approaches either require costly retraining or suffer steep quality degradation when aggressively dropping context.

--- Method / Approach ---
  - Identifies that attention scores follow a power-law distribution where "Heavy Hitters" (H2) dominate attention value [Section: Method]
  - Formulates KV cache eviction as a dynamic submodular maximization problem with theoretical approximation guarantees [Section: Method]
  - Implements an eviction policy combining a local sliding window (recent tokens) with a global pool of heavy hitters (accumulated attention scores) [Section: Method]
  - Evaluates across OPT, LLaMA, and GPT-NeoX architectures without requiring model fine-tuning or weight retraining [Section: Method]
  - Integrates with leading inference backends (DeepSpeed Zero-Inference, Hugging Face Accelerate, FlexGen) [Section: Method]

--- Key Results / Claims ---
  - Retaining just 20% of the KV cache achieves comparable perplexity and downstream accuracy to full-cache baselines [Section: Results]
  - Improves inference throughput up to 29x over DeepSpeed Zero-Inference and Accelerate, and up to 3x over FlexGen on OPT-6.7B and OPT-30B [Section: Results]
  - Reduces inference latency by up to 1.9x at equivalent batch sizes [Section: Results]
  - Demonstrates sustained generation coherence on sequences up to 2,000+ tokens with bounded memory footprint [Section: Results]

--- Limitations ---
  - Theoretical guarantees depend on submodularity conditions that may be sensitive to outlier token activation patterns [Section: Limitations]
  - Tracking cumulative attention scores introduces minor tracking overhead during attention matrix computation [Section: Limitations]
  - Primary evaluation is on decoder-only autoregressive models; sparse attention architectures and mixture-of-experts were not tested [Section: Limitations]

--- Suggested Follow-up Questions ---
  - How does H2O determine the optimal ratio between recent window size and heavy-hitter budget?
  - Can H2O be composed with 4-bit/8-bit KV quantization for orthogonal memory reductions?
  - Does the heavy-hitter token distribution change significantly between prefilling and decoding phases?
==============================================================================

(Briefing saved to briefing_2306_14048.json; will be updated with QA history on exit.)

Ask follow-up questions about the paper (type 'exit' or 'quit' to stop):

Q: How does H2O decide which tokens to evict from the KV cache?
A: In the Method section, H2O tracks cumulative attention scores for each token across generation steps to identify "Heavy Hitters" (H2). The eviction algorithm maintains a fixed memory budget partitioned between a sliding window of recent tokens (for local syntactic context) and the highest-accumulated heavy hitter tokens. When new tokens arrive and the budget is exceeded, tokens with the lowest cumulative attention scores outside the recent window are dropped.
  Sources: [method | 2306_14048_2], [method | 2306_14048_3]

Q: Does H2O require fine-tuning or modifying the pre-trained model weights?
A: In the Introduction and Method sections, the paper emphasizes that H2O is a training-free, drop-in KV cache eviction policy. It operates entirely during runtime inference on unmodified weights of pre-trained models such as OPT, LLaMA, and GPT-NeoX.
  Sources: [introduction | 2306_14048_0], [method | 2306_14048_1]

Q: Does H2O support convolutional neural networks like ResNet-50?
A: I couldn't find an answer to this in the retrieved parts of the paper. It may not be covered, or may be in a section that wasn't parsed. [low confidence / ungrounded in paper]

Q: exit
Final briefing + QA history saved to briefing_2306_14048.json
```

---

## Project Structure

```
akari/
├── main.py                    # CLI entrypoint (Rich UI, interactive selection, session resumption)
├── requirements.txt            # Constrained core dependencies
├── requirements-dev.txt        # Development, testing, and CI tooling (pytest, ruff, mypy)
├── pyproject.toml              # Project metadata, optional extras ([groq], [dev]), tool configs
├── Makefile                    # Developer workflow (test, eval, lint, format, typecheck, clean)
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI matrix (Python 3.10, 3.11, 3.12)
├── agent/
│   ├── config.py               # Durable, configurable data paths (~/.cache/akari)
│   ├── embedding.py            # Local embedder factory + deterministic offline test mock
│   ├── evaluation.py           # Grounding & retrieval benchmark harness
│   ├── state.py                # GraphState + all typed dataclasses (shared state schema)
│   ├── graph.py                # Sequential pipeline runner with self-correction loops
│   ├── llm_client.py           # Swappable LLM interface (Gemini/Groq/Ollama)
│   ├── session.py              # SQLite session store with WAL mode & busy timeout retry
│   ├── ui.py                   # Rich terminal UI (banners, tables, formatted briefings)
│   └── nodes/
│       ├── query_understanding.py
│       ├── arxiv_retrieval.py
│       ├── ranking.py
│       ├── fetch_parse.py
│       ├── chunk_embed.py
│       ├── summarize.py
│       └── qa.py
└── tests/
    ├── test_agent.py           # 100% offline unit tests (deterministic embeddings, isolation, mocks)
    └── test_evaluation.py      # Automated regression tests for grounding and retrieval benchmark
```

---

## Retrieval, Citation & Grounding Evaluation Benchmark

To move beyond informal heuristics and prevent confident but unsupported outputs, Akari includes an automated evaluation harness (`agent/evaluation.py` and `tests/test_evaluation.py`) that quantitatively benchmarks retrieval quality, citation accuracy for both answers and summaries, fact faithfulness, and negative rejection of out-of-scope queries.

The evaluation suite tests 10 QA benchmark queries (covering KV-cache eviction mechanisms, theoretical bounds, empirical accuracy, baseline throughput speedup, and compute overhead caveats, alongside adversarial unsupported queries on unrelated architectures and funding) plus a summary citation fidelity suite verifying that every briefing point cites its source section.

### Benchmark Results

| Evaluation Dimension | Metric | Benchmark Score | Target Threshold | Status |
|:---------------------|:-------|:----------------|:-----------------|:-------|
| **Retrieval Relevance** | Hit Rate @ 1 (Hit@1) | **83.3%** | $\ge 75.0\%$ | Passed |
| **Retrieval Relevance** | Hit Rate @ 3 (Hit@3) | **100.0%** | $\ge 85.0\%$ | Passed |
| **Retrieval Relevance** | Hit Rate @ 5 (Hit@5) | **100.0%** | $\ge 90.0\%$ | Passed |
| **Retrieval Relevance** | Mean Reciprocal Rank (MRR) | **0.917** | $\ge 0.800$ | Passed |
| **Citation Verification** | QA Section Citation Accuracy | **100.0%** | $\ge 80.0\%$ | Passed |
| **Citation Verification** | QA Citation Precision | **100.0%** | $\ge 80.0\%$ | Passed |
| **Citation Verification** | Summary Citation Attribution Rate | **100.0%** | $\ge 85.0\%$ | Passed |
| **Grounding & Refusal** | Refusal Precision (Out-of-Scope) | **100.0%** | $\ge 90.0\%$ | Passed |
| **Grounding & Refusal** | Refusal Recall (Out-of-Scope) | **100.0%** | $\ge 90.0\%$ | Passed |
| **Grounding & Refusal** | Refusal F1 Score | **100.0%** | $\ge 90.0\%$ | Passed |
| **Answer Quality** | Fact Coverage & Completeness | **100.0%** | $\ge 80.0\%$ | Passed |
| **Answer Quality** | Grounded Faithfulness | **100.0%** | $\ge 85.0\%$ | Passed |

Run the benchmark locally:

```bash
make eval
# or directly:
python -m agent.evaluation
```

---

## Testing & Quality Assurance

### Deterministic Offline Test Suite

The test suite runs **100% offline with zero external network access or model downloads**. Embedding functions and vector indexing are decoupled through `agent/embedding.py`, using token-feature hashing in tests to guarantee deterministic execution in under 4 seconds without touching Hugging Face Hub:

```bash
# Run tests via Makefile:
make test

# Or directly with pytest:
pytest tests/ -v

# Or with standard library unittest:
python -m unittest discover tests
```

### Static Analysis, Linting & Type Checking

Code quality and typing standards are enforced via `ruff` and `mypy`:

```bash
make lint       # Ruff lint check (selects E, F, W, I, B)
make format     # Ruff formatting
make typecheck  # Mypy static type verification across agent/ and tests/
```

### Continuous Integration (CI)

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request across a multi-version Python matrix:
- **Supported Python Matrix**: Python 3.10, 3.11, and 3.12
- **Stages**: Dependency installation $\rightarrow$ Ruff linting & format validation $\rightarrow$ Mypy type-checking $\rightarrow$ Offline unit test execution $\rightarrow$ Retrieval & Grounding benchmark execution

---

## Technical Interview Guide (Talking Points)

If an interviewer asks you about this project, here are the real, honest engineering decisions and tradeoffs made, explained with technical clarity:

### 1. Why build your own state graph instead of using LangChain, LangGraph, or Pydantic?
> *"LangGraph, CrewAI, and heavy Pydantic stacks are popular, but for an engineering pipeline of this scope, importing massive frameworks introduces hidden abstractions, startup latency, and dependency lock-in. I chose pure Python dataclasses (`GraphState`) and explicit, pure functional node contracts. This makes state inspectable, eliminates runtime overhead, allows fast deterministic offline testing with simple mocks, and prevents framework deprecation headaches."*

### 2. How did you solve data isolation and concurrency risks across runs?
> *"In earlier iterations, Chroma collections and downloaded PDFs used shared names and ephemeral `/tmp` storage, which created race conditions if multiple queries ran concurrently or when the OS cleaned `/tmp`. I re-architected persistence around durable directories (`~/.cache/akari`) with run-isolated collection names (`paper_<id>_<uuid>`). Downloads now write to temporary files before atomically replacing the target (`os.replace`), and SQLite sessions use Write-Ahead Logging (`WAL`) with busy timeout retry locks to prevent concurrent write collisions."*

### 3. How do you prevent hallucinations in the QA loop, and how is it verified?
> *"Hallucination mitigation uses a multi-layered grounding strategy: retrieval-bounded context (the model only sees top-k chunks), strict refusal instructions with a deterministic refusal token (`NOT_IN_PAPER`), and cosine distance guardrails. Critically, rather than relying solely on heuristics, we built a reproducible evaluation benchmark (`agent/evaluation.py`) that quantitatively tests retrieval recall (Hit@3=100%, MRR=0.917), section citation accuracy (100%), fact coverage (100%), and negative rejection for out-of-scope questions (100% Refusal F1)."*

### 4. How did you handle PDF parsing challenges (two columns, math, figures)?
> *"Standard PDF extractors often read across two-column scientific papers horizontally, garbling sentences together. I addressed this with PyMuPDF block geometry: inspecting bounding box coordinates `(x0, y0)` and sorting blocks by horizontal column boundaries before stitching text. Furthermore, because some PDFs are scanned or fail layout extraction, I added an automated fallback to ar5iv clean HTML before gracefully degrading to abstract-only mode if text density is below a strict threshold."*

### 5. Why use a local embedding model instead of OpenAI/Gemini embeddings?
> *"Running `all-MiniLM-L6-v2` locally via `sentence-transformers` ensures vector indexing runs free on any machine with zero API egress costs, no rate limits, and consistent embedding dimensions across sessions. For testing and CI, the architecture decouples vector generation through an embedding abstraction, allowing 100% offline testing without downloading model weights."*

### 6. How do you prevent confident but unsupported summaries and answers, and verify citation accuracy?
> *"A paper assistant that doesn't show where each claim originated will produce confident but unsupported summaries. We addressed this with end-to-end source attribution: the summarization node binds every methodology bullet, result, and limitation directly to extracted section digests with mandatory `[Section: <Name>]` citations and structured `section_citations` mappings. In QA, every response surfaces its source sections and underlying chunk IDs. Crucially, we don't cite automated quality as a strength without verifying it: our CI workflow runs a deterministic benchmark across Python 3.10, 3.11, and 3.12, verifying citation accuracy, fact coverage, and refusal on adversarial out-of-scope questions."*

