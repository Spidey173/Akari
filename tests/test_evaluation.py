"""
Unit tests for the Retrieval, Citation & Grounding Evaluation Benchmark.
Runs 100% offline with zero external network or LLM API dependencies.
"""

from __future__ import annotations

import os
import unittest

os.environ["AKARI_OFFLINE_TEST"] = "1"

from agent.embedding import MockEmbeddingFunction, get_chroma_client
from agent.evaluation import (
    BENCHMARK_QUESTIONS,
    BenchmarkDeterministicLLM,
    EvaluationRunner,
)


class TestEvaluationBenchmark(unittest.TestCase):
    def setUp(self) -> None:
        self.client = get_chroma_client(ephemeral=True)
        self.embed_fn = MockEmbeddingFunction()
        self.llm = BenchmarkDeterministicLLM()
        self.runner = EvaluationRunner(self.client, self.embed_fn, self.llm)

    def test_benchmark_metrics_pass_targets(self) -> None:
        metrics = self.runner.run_benchmark(BENCHMARK_QUESTIONS)

        self.assertEqual(metrics.total_samples, 10)
        self.assertEqual(metrics.supported_samples, 6)
        self.assertEqual(metrics.unsupported_samples, 4)

        # Retrieval Relevance
        self.assertGreaterEqual(metrics.hit_at_1, 0.75)
        self.assertGreaterEqual(metrics.hit_at_3, 0.85)
        self.assertGreaterEqual(metrics.hit_at_5, 0.90)
        self.assertGreaterEqual(metrics.mrr, 0.80)

        # Citation Verification
        self.assertGreaterEqual(metrics.citation_accuracy, 0.80)
        self.assertGreaterEqual(metrics.citation_precision, 0.80)
        self.assertGreaterEqual(metrics.citation_recall, 0.80)
        self.assertGreaterEqual(metrics.summary_citation_rate, 0.85)

        # Grounding & Refusal
        self.assertGreaterEqual(metrics.refusal_precision, 0.90)
        self.assertGreaterEqual(metrics.refusal_recall, 0.90)
        self.assertGreaterEqual(metrics.refusal_f1, 0.90)

        # Answer Quality
        self.assertGreaterEqual(metrics.faithfulness_score, 0.85)
        self.assertGreaterEqual(metrics.fact_coverage_score, 0.80)


if __name__ == "__main__":
    unittest.main()
