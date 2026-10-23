import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ["AKARI_OFFLINE_TEST"] = "1"

from agent.embedding import MockEmbeddingFunction, set_chroma_client, set_embedding_function
from agent.graph import run_pipeline
from agent.llm_client import LLMClient
from agent.nodes import chunk_embed, fetch_parse, qa, query_understanding, ranking, summarize
from agent.session import list_sessions, load_session, save_session
from agent.state import Briefing, GraphState, NodeStatus, PaperMetadata


class MockLLM(LLMClient):
    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.prompts: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> str:
        self.prompts.append((prompt, system))
        return self.response


class BaseOfflineTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.chroma_dir = os.path.join(self.test_dir, "chroma")
        self.pdf_dir = os.path.join(self.test_dir, "pdfs")
        os.makedirs(self.chroma_dir, exist_ok=True)
        os.makedirs(self.pdf_dir, exist_ok=True)
        set_embedding_function(MockEmbeddingFunction())
        set_chroma_client(None)

    def tearDown(self):
        set_embedding_function(None)
        set_chroma_client(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)


class TestQueryUnderstanding(unittest.TestCase):
    def test_explicit_arxiv_id(self):
        state = GraphState(raw_query="2401.12345")
        state = query_understanding.run(state)
        self.assertEqual(state.intent, "specific_paper")
        self.assertEqual(state.explicit_arxiv_id, "2401.12345")
        self.assertEqual(state.status, NodeStatus.OK)

    def test_explicit_arxiv_url(self):
        state = GraphState(raw_query="https://arxiv.org/abs/2305.18290v1")
        state = query_understanding.run(state)
        self.assertEqual(state.intent, "specific_paper")
        self.assertEqual(state.explicit_arxiv_id, "2305.18290")
        self.assertEqual(state.status, NodeStatus.OK)

    def test_topic_search(self):
        state = GraphState(raw_query="recent work on KV-cache compression for LLMs")
        state = query_understanding.run(state)
        self.assertEqual(state.intent, "topic_search")
        self.assertIsNone(state.explicit_arxiv_id)
        self.assertEqual(state.search_query, "recent work on KV-cache compression for LLMs")
        self.assertEqual(state.status, NodeStatus.OK)

    def test_empty_query(self):
        state = GraphState(raw_query="")
        state = query_understanding.run(state)
        self.assertEqual(state.status, NodeStatus.FAILED)
        self.assertTrue(any("Empty query" in err for err in state.errors))


class TestRanking(unittest.TestCase):
    def test_specific_paper_skips_ranking(self):
        paper = PaperMetadata(
            arxiv_id="2401.12345",
            title="A Great Paper",
            authors=["Alice"],
            abstract="Summary",
            pdf_url="http://example.com/pdf",
            published="2024-01-01",
            categories=["cs.AI"],
        )
        state = GraphState(
            intent="specific_paper",
            candidates=[paper],
        )
        state = ranking.run(state)
        self.assertEqual(state.selected_paper, paper)
        self.assertEqual(state.status, NodeStatus.OK)

    def test_topic_ranking_by_overlap(self):
        p1 = PaperMetadata(
            arxiv_id="1",
            title="Unrelated Paper on Robotics",
            authors=["Bob"],
            abstract="Robotics manipulation in warehouse environments.",
            pdf_url="http://example.com/1",
            published="2024-01-01",
            categories=["cs.RO"],
        )
        p2 = PaperMetadata(
            arxiv_id="2",
            title="Efficient KV-Cache Compression for Large Language Models",
            authors=["Carol"],
            abstract="We propose a novel KV cache pruning technique for transformer inference.",
            pdf_url="http://example.com/2",
            published="2024-01-01",
            categories=["cs.CL"],
        )
        state = GraphState(
            intent="topic_search",
            search_query="KV-cache compression LLMs",
            candidates=[p1, p2],
        )
        state = ranking.run(state)
        assert state.selected_paper is not None
        self.assertEqual(state.selected_paper.arxiv_id, "2")
        self.assertGreater(p2.score, p1.score)


class TestFetchParse(unittest.TestCase):
    def test_section_splitting(self):
        sample_text = (
            "Title and frontmatter\n\n"
            "Introduction\n"
            "Here is the intro text.\n\n"
            "Methodology\n"
            "Here is the method text.\n\n"
            "Results\n"
            "We achieve 95% accuracy.\n\n"
            "Conclusion\n"
            "Final remarks.\n"
        )
        sections = fetch_parse._split_sections(sample_text)
        self.assertIn("introduction", sections)
        self.assertIn("methodology", sections)
        self.assertIn("results", sections)
        self.assertIn("conclusion", sections)
        self.assertEqual(sections["results"], "We achieve 95% accuracy.")

    def test_degraded_parsing_fallback(self):
        paper = PaperMetadata(
            arxiv_id="2401.99999",
            title="Scanned Paper",
            authors=["Dave"],
            abstract="This paper explores scanned documents.",
            pdf_url="http://example.com/scanned.pdf",
            published="2024-01-01",
            categories=["cs.CV"],
        )
        state = GraphState(
            selected_paper=paper,
        )
        with (
            patch.object(fetch_parse, "_download_pdf", return_value="/fake/path.pdf"),
            patch.object(
                fetch_parse, "_extract_text_layout_aware", return_value=("Very short text", 5)
            ),
            patch.object(fetch_parse, "_fetch_ar5iv_html", return_value=None),
        ):
            state = fetch_parse.run(state)

        self.assertTrue(state.parse_degraded)
        self.assertEqual(state.full_text, paper.abstract)
        self.assertEqual(state.sections, {"abstract": paper.abstract})
        self.assertEqual(state.status, NodeStatus.OK)

    def test_download_atomic_concurrency_safety(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resp = MagicMock()
            mock_resp.content = b"%PDF-1.4 Mock PDF binary content"
            with patch("requests.get", return_value=mock_resp):
                path = fetch_parse._download_pdf(
                    "http://example.com/test.pdf", "2401.12345", download_dir=tmpdir
                )
                self.assertTrue(os.path.exists(path))
                self.assertEqual(os.path.basename(path), "2401.12345.pdf")
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), mock_resp.content)


class TestChunking(BaseOfflineTestCase):
    def test_sentence_aware_chunks(self):
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = chunk_embed._sentence_aware_chunks(text, size=45, overlap=10)
        self.assertTrue(len(chunks) > 1)
        for c in chunks:
            self.assertFalse(c.endswith(" sentenc"))

    def test_collection_isolation_between_runs(self):
        paper = PaperMetadata(
            arxiv_id="2401.12345",
            title="Sample Paper",
            authors=["Alice"],
            abstract="Abstract text",
            pdf_url="http://example.com/pdf",
            published="2024-01-01",
            categories=["cs.AI"],
        )
        state1 = GraphState(
            selected_paper=paper,
            sections={"method": "Sentence one for testing. Sentence two for testing."},
        )
        state1 = chunk_embed.run(state1, chroma_path=self.chroma_dir)
        self.assertEqual(state1.status, NodeStatus.OK)
        col1 = state1.collection_name

        state2 = GraphState(
            selected_paper=paper,
            sections={"method": "Sentence one for testing. Sentence two for testing."},
        )
        state2 = chunk_embed.run(state2, chroma_path=self.chroma_dir)
        self.assertEqual(state2.status, NodeStatus.OK)
        col2 = state2.collection_name

        # Ensure collection names are isolated to prevent cross-run interference
        assert col1 is not None and col2 is not None
        self.assertNotEqual(col1, col2)
        self.assertTrue(col1.startswith("paper_2401_12345_"))
        self.assertTrue(col2.startswith("paper_2401_12345_"))


class TestSummarize(unittest.TestCase):
    def test_extract_json_clean(self):
        raw = '{"summary": "Test summary", "problem_statement": "Problem", "method_bullets": ["m1"], "key_results": ["r1"], "limitations": ["l1"], "followup_questions": ["q1"]}'
        data = summarize._extract_json(raw)
        self.assertEqual(data["summary"], "Test summary")

    def test_extract_json_markdown_wrapped(self):
        raw = 'Here is the briefing:\n```json\n{"summary": "Test summary", "problem_statement": "Problem", "method_bullets": [], "key_results": [], "limitations": ["l1", "l2"], "followup_questions": []}\n```'
        data = summarize._extract_json(raw)
        self.assertEqual(data["summary"], "Test summary")

    def test_limitations_enforced(self):
        raw = json.dumps(
            {
                "summary": "Summary text",
                "problem_statement": "Problem",
                "method_bullets": ["method 1"],
                "key_results": ["result 1"],
                "limitations": [],  # Empty limitations returned by model
                "followup_questions": ["question 1"],
            }
        )
        llm = MockLLM(response=raw)
        paper = PaperMetadata(
            arxiv_id="2401.00001",
            title="Sample Paper",
            authors=["Author A"],
            abstract="Abstract text",
            pdf_url="http://example.com/pdf",
            published="2024-01-01",
            categories=["cs.AI"],
        )
        state = GraphState(selected_paper=paper, full_text="Full text content here.")
        state = summarize.run(state, llm)
        self.assertEqual(state.status, NodeStatus.OK)
        self.assertIsNotNone(state.briefing)
        assert state.briefing is not None
        self.assertTrue(len(state.briefing.limitations) > 0)


class TestQA(BaseOfflineTestCase):
    def test_format_context(self):
        chunks = [
            ("id_1", "method", "We used Adam optimizer."),
            ("id_2", "results", "Accuracy reached 92%."),
        ]
        formatted = qa._format_context(chunks)
        self.assertIn("[id_1 | section: method]", formatted)
        self.assertIn("Accuracy reached 92%", formatted)

    def test_refusal_marker_handling(self):
        llm = MockLLM(response="NOT_IN_PAPER")
        state = GraphState(collection_name="test_col", qa_history=[])
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["c1"]],
            "documents": [["Some unrelated text about climate change."]],
            "metadatas": [[{"section": "intro"}]],
            "distances": [[0.3]],
        }
        exchange = qa.answer_question(
            state, llm, "What was the learning rate?", collection=mock_collection
        )

        self.assertFalse(exchange.grounded)
        self.assertIn("couldn't find an answer", exchange.answer)


class TestSessionCache(unittest.TestCase):
    def test_save_and_load_session(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            db_path = tmp.name
            paper = PaperMetadata(
                arxiv_id="2401.12345",
                title="Caching Test Paper",
                authors=["Test Author"],
                abstract="Abstract",
                pdf_url="http://example.com/test.pdf",
                published="2024-01-01",
                categories=["cs.AI"],
            )
            briefing = Briefing(
                title="Caching Test Paper",
                authors=["Test Author"],
                arxiv_id="2401.12345",
                published="2024-01-01",
                link="http://arxiv.org/abs/2401.12345",
                summary="A test summary",
                problem_statement="A problem statement",
                method_bullets=["Bullet 1"],
                key_results=["Result 1"],
                limitations=["Limitation 1"],
                followup_questions=["Question 1"],
            )
            state = GraphState(
                selected_paper=paper,
                briefing=briefing,
                collection_name="paper_2401_12345",
            )
            save_session(state, db_path=db_path)

            sessions = list_sessions(db_path=db_path)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["arxiv_id"], "2401.12345")

            loaded = load_session("2401.12345", db_path=db_path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            assert loaded.selected_paper is not None
            assert loaded.briefing is not None
            self.assertEqual(loaded.selected_paper.title, "Caching Test Paper")
            self.assertEqual(loaded.collection_name, "paper_2401_12345")
            self.assertEqual(loaded.briefing.summary, "A test summary")


class TestPipeline(BaseOfflineTestCase):
    def test_pipeline_halts_on_empty_query(self):
        llm = MockLLM()
        state = run_pipeline("", llm)
        self.assertEqual(state.status, NodeStatus.FAILED)
        self.assertTrue(len(state.errors) > 0)
        self.assertIn("query_understanding", state.errors[0])

    def test_pipeline_end_to_end_mocked(self):
        mock_entry = MagicMock()
        mock_entry.id = "http://arxiv.org/abs/2401.12345v1"
        mock_entry.title = "Mock Paper Title"
        mock_entry.authors = [MagicMock(name="Test Author")]
        mock_entry.summary = "This is a mock abstract."
        mock_entry.links = [MagicMock(href="http://example.com/pdf", title="pdf")]
        mock_entry.published = "2024-01-01"
        mock_entry.tags = [{"term": "cs.AI"}]

        raw_briefing = json.dumps(
            {
                "summary": "Mock summary for paper",
                "problem_statement": "Mock problem",
                "method_bullets": ["Step 1", "Step 2"],
                "key_results": ["Result 1"],
                "limitations": ["Limitation 1", "Limitation 2"],
                "followup_questions": ["Question 1"],
            }
        )
        llm = MockLLM(response=raw_briefing)

        with (
            patch("agent.nodes.arxiv_retrieval._query_arxiv", return_value=[mock_entry]),
            patch("agent.nodes.fetch_parse._download_pdf", return_value="/tmp/mock.pdf"),
            patch(
                "agent.nodes.fetch_parse._extract_text_layout_aware",
                return_value=(
                    "Abstract\nMock paper body with plenty of words to exceed threshold...",
                    1,
                ),
            ),
        ):
            state = run_pipeline("2401.12345", llm)

        self.assertEqual(state.status, NodeStatus.OK)
        self.assertIsNotNone(state.briefing)
        assert state.briefing is not None
        self.assertEqual(state.briefing.arxiv_id, "2401.12345")
        self.assertEqual(state.briefing.title, "Mock Paper Title")
        self.assertTrue(len(state.chunks) > 0)
        self.assertIsNotNone(state.collection_name)

    def test_pipeline_query_reformulation_on_zero_results(self):
        """Tests that the agentic loop catches zero search results on noisy queries and reformulates."""
        mock_entry = MagicMock()
        mock_entry.id = "http://arxiv.org/abs/2401.55555v1"
        mock_entry.title = "Sparse Attention Paper"
        mock_entry.authors = [MagicMock(name="Author")]
        mock_entry.summary = "Abstract content"
        mock_entry.links = [MagicMock(href="http://example.com/pdf", title="pdf")]
        mock_entry.published = "2024-01-01"
        mock_entry.tags = [{"term": "cs.AI"}]

        query_calls = []

        def mock_query(q, max_results=5):
            query_calls.append(q)
            if len(query_calls) == 1:
                return []
            return [mock_entry]

        raw_briefing = json.dumps(
            {
                "summary": "Mock summary",
                "problem_statement": "Mock problem",
                "method_bullets": ["Step 1"],
                "key_results": ["Result 1"],
                "limitations": ["Limitation 1", "Limitation 2"],
                "followup_questions": ["Question 1"],
            }
        )
        llm = MockLLM(response=raw_briefing)

        with (
            patch("agent.nodes.arxiv_retrieval._query_arxiv", side_effect=mock_query),
            patch("agent.nodes.fetch_parse._download_pdf", return_value="/tmp/mock.pdf"),
            patch(
                "agent.nodes.fetch_parse._extract_text_layout_aware",
                return_value=(
                    "Abstract\nMock paper body with plenty of words to exceed threshold...",
                    1,
                ),
            ),
        ):
            state = run_pipeline("recent work on sparse attention", llm)

        self.assertEqual(state.status, NodeStatus.OK)
        self.assertGreaterEqual(len(query_calls), 2)
        self.assertIn("sparse attention", query_calls[-1])
        self.assertNotIn("recent", query_calls[-1])


if __name__ == "__main__":
    unittest.main()
