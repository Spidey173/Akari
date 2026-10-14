# Akari (明かり) — Autonomous arXiv Paper Digest & QA Agent

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
                    ┌─────────────────────┐
  user query   ───▶ │ Query Understanding  │  (regex: arXiv ID vs topic)
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │   arXiv Retrieval    │  (official Atom API)
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Selection/Ranking   │  (lexical overlap scoring;
                    │  (skipped if exact   │   skipped for direct ID lookup)
                    │   ID was given)      │
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Fetch & Parse      │  (PyMuPDF; degrades to
                    │                      │   abstract-only on failure)
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Chunk & Embed       │  (sentence-aware chunking,
                    │                      │   per-section; Chroma + MiniLM)
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Summarize          │  (LLM call → structured JSON
                    │                      │   → Briefing dataclass)
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
  user questions ──▶│      QA Loop         │  (RAG: retrieve top-k chunks →
       ◀────────────│   (interactive)      │   LLM answers only from those
       answers      └──────────────────────┘   chunks, or explicitly declines)
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

### Persistence & Session Resumption

State is **in-memory during a single run** (the `GraphState` object lives for
the process's lifetime), while index vectors and structured session data are
persisted independently for instant resumption:

- **Vector Store Persistence**: Vector chunks are persisted in a local Chroma
  store (`/tmp/akari_chroma`), surviving process termination so that future
  QA sessions do not need to re-parse or re-embed the paper.
- **SQLite Session Cache**: Paper briefings and QA history are automatically
  stored in a SQLite database (`~/.cache/akari/sessions.db`). Running
  `python main.py --resume <arxiv_id>` restores the full executive briefing and
  connects immediately to the interactive QA loop in <1 second without re-fetching.
- **JSON Export**: An exportable JSON file (`briefing_<id>.json`) is generated
  for programmatic consumption or integration into downstream workflows.

---

## Design Decisions & Tradeoffs

**Lightweight Orchestration over Heavy Frameworks.** Rather than adding heavy
framework dependencies (e.g. LangGraph or CrewAI) for what is fundamentally a
structured directed pipeline, Akari uses a transparent Python dataclass runner
with explicit node contracts:
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
3. **Fine-Grained Briefing Citations**: Map individual briefing bullet points
   directly to source chunk IDs, mirroring the section citations in QA.
4. **Automated Evaluation Benchmark**: A reproducible evaluation harness using
   ground-truth Q&A pairs (via Ragas) to continuously quantify retrieval recall
   and refusal precision.

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
  - Identifies that attention scores follow a power-law distribution where "Heavy Hitters" (H2) dominate attention value.
  - Formulates KV cache eviction as a dynamic submodular maximization problem with theoretical approximation guarantees.
  - Implements an eviction policy combining a local sliding window (recent tokens) with a global pool of heavy hitters (accumulated attention scores).
  - Evaluates across OPT, LLaMA, and GPT-NeoX architectures without requiring model fine-tuning or weight retraining.
  - Integrates with leading inference backends (DeepSpeed Zero-Inference, Hugging Face Accelerate, FlexGen).

--- Key Results / Claims ---
  - Retaining just 20% of the KV cache achieves comparable perplexity and downstream accuracy to full-cache baselines.
  - Improves inference throughput up to 29x over DeepSpeed Zero-Inference and Accelerate, and up to 3x over FlexGen on OPT-6.7B and OPT-30B.
  - Reduces inference latency by up to 1.9x at equivalent batch sizes.
  - Demonstrates sustained generation coherence on sequences up to 2,000+ tokens with bounded memory footprint.

--- Limitations ---
  - Theoretical guarantees depend on submodularity conditions that may be sensitive to outlier token activation patterns.
  - Tracking cumulative attention scores introduces minor tracking overhead during attention matrix computation.
  - Primary evaluation is on decoder-only autoregressive models; sparse attention architectures and mixture-of-experts were not tested.

--- Suggested Follow-up Questions ---
  - How does H2O determine the optimal ratio between recent window size and heavy-hitter budget?
  - Can H2O be composed with 4-bit/8-bit KV quantization for orthogonal memory reductions?
  - Does the heavy-hitter token distribution change significantly between prefilling and decoding phases?
==============================================================================

(Briefing saved to briefing_2306_14048.json; will be updated with QA history on exit.)

Ask follow-up questions about the paper (type 'exit' or 'quit' to stop).

Q: How does H2O decide which tokens to evict from the KV cache?
A: In the Method section, H2O tracks cumulative attention scores for each token across generation steps to identify "Heavy Hitters" (H2). The eviction algorithm maintains a fixed memory budget partitioned between a sliding window of recent tokens (for local syntactic context) and the highest-accumulated heavy hitter tokens. When new tokens arrive and the budget is exceeded, tokens with the lowest cumulative attention scores outside the recent window are dropped.

Q: Does H2O require fine-tuning or modifying the pre-trained model weights?
A: In the Introduction and Method sections, the paper emphasizes that H2O is a training-free, drop-in KV cache eviction policy. It operates entirely during runtime inference on unmodified weights of pre-trained models such as OPT, LLaMA, and GPT-NeoX.

Q: Does H2O support convolutional neural networks like ResNet-50?
A: I couldn't find an answer to this in the retrieved parts of the paper. It may not be covered, or may be in a section that wasn't parsed.  [low confidence / not clearly in paper]

Q: exit
Final briefing + QA history saved to briefing_2306_14048.json
```

---

## Project Structure

```
akari/
├── main.py                    # CLI entrypoint (Rich UI, interactive selection, session resumption)
├── requirements.txt
├── .env.example
├── agent/
│   ├── state.py                # GraphState + all dataclasses (shared state schema)
│   ├── graph.py                 # sequential pipeline runner (the "graph")
│   ├── llm_client.py            # swappable LLM interface (Gemini/Groq/Ollama)
│   ├── session.py               # SQLite session store for instant resumption (--resume)
│   ├── ui.py                    # Rich terminal UI (banners, tables, formatted briefings)
│   └── nodes/
│       ├── query_understanding.py
│       ├── arxiv_retrieval.py
│       ├── ranking.py
│       ├── fetch_parse.py
│       ├── chunk_embed.py
│       ├── summarize.py
│       └── qa.py
└── tests/
    └── test_agent.py           # unit tests for graph nodes, session cache, fallbacks, grounding
```

---

## Testing

An automated unit test suite is included in `tests/test_agent.py`, covering:
- Intent parsing for arXiv IDs, URLs, and natural language topic queries
- Keyword-overlap ranking score calculations
- Heuristic section splitting and scanned PDF degradation fallback
- Sentence-aware chunking with overlap boundaries
- Robust JSON parsing and fallback extraction from LLM outputs
- QA grounding checks and refusal token (`NOT_IN_PAPER`) handling
- End-to-end pipeline execution and early halt on failures

To run the test suite:

```bash
pytest tests/
# or with standard library unittest:
python -m unittest discover tests
```

All unit tests run offline using mocked responses without requiring an external LLM API key or network access. Run `python main.py` on a machine with normal internet access to exercise the full live pipeline end-to-end against the arXiv API.

