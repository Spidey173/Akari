"""
Unit tests for the Retrieval & Grounding Evaluation Benchmark.
Runs 100% offline with zero external network or LLM API dependencies.
"""

import os
import unittest

os.environ["AKARI_OFFLINE_TEST"] = "1"

from agent.embedding import MockEmbeddingFunction, get_chroma_client
from agent.evaluation import BENCHMARK_QUESTIONS, EvaluationRunner
from agent.llm_client import LLMClient
from agent.nodes import qa


class DeterministicEvalLLM(LLMClient):
    """Offline deterministic LLM emulator for grounding benchmark tests."""

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> str:
        q_text = prompt.lower()
        for line in prompt.split("\n"):
            if line.lower().startswith("question:"):
                q_text = line.lower()
                break

        if "resnet" in q_text or "quantum" in q_text or "venture" in q_text:
            return qa.REFUSAL_MARKER
        if "eviction" in q_text or "selected" in q_text:
            return (
                "In the Method section, the authors explain that tokens are selected for "
                "eviction based on cumulative attention scores to identify Heavy Hitters, "
                "evicting tokens outside a local sliding window."
            )
        if "accuracy" in q_text:
            return (
                "In the Results section, empirical evaluation demonstrates that keeping "
                "20% of the KV cache maintains 99.2% of downstream zero-shot accuracy."
            )
        if "throughput" in q_text:
            return (
                "In the Results section, the paper notes that inference throughput "
                "increases by up to 2.8x compared to FlexGen."
            )
        if "computational overhead" in q_text or "limitation" in q_text or "overhead" in q_text:
            return (
                "In the Limitations section, the paper highlights that tracking "
                "cumulative attention scores adds a 3-5% latency penalty during prefilling."
            )
        return qa.REFUSAL_MARKER


class TestEvaluationBenchmark(unittest.TestCase):
    def setUp(self):
        self.client = get_chroma_client(ephemeral=True)
        self.embed_fn = MockEmbeddingFunction()
        self.llm = DeterministicEvalLLM()
        self.runner = EvaluationRunner(self.client, self.embed_fn, self.llm)

    def test_benchmark_metrics_pass_targets(self):
        metrics = self.runner.run_benchmark(BENCHMARK_QUESTIONS)

        self.assertEqual(metrics.total_samples, 7)
        self.assertEqual(metrics.supported_samples, 4)
        self.assertEqual(metrics.unsupported_samples, 3)

        # Retrieval Relevance
        self.assertGreaterEqual(metrics.hit_at_1, 0.75)
        self.assertGreaterEqual(metrics.hit_at_3, 0.85)
        self.assertGreaterEqual(metrics.mrr, 0.80)

        # Citation Accuracy
        self.assertGreaterEqual(metrics.citation_accuracy, 0.80)

        # Refusal & Grounding
        self.assertGreaterEqual(metrics.refusal_precision, 0.90)
        self.assertGreaterEqual(metrics.refusal_recall, 0.90)
        self.assertGreaterEqual(metrics.refusal_f1, 0.90)

        # Faithfulness
        self.assertGreaterEqual(metrics.faithfulness_score, 0.85)


if __name__ == "__main__":
    unittest.main()
