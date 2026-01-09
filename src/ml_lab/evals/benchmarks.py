"""Evaluation benchmarks for model testing."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvalResult:
    """Result from running an evaluation."""

    benchmark_name: str
    model_name: str
    score: float
    max_score: float
    metrics: dict[str, float] = field(default_factory=dict)
    samples_evaluated: int = 0
    errors: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: list[dict[str, Any]] = field(default_factory=list)


class Benchmark(ABC):
    """Abstract base class for evaluation benchmarks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Benchmark description."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        model_path: str,
        num_samples: int | None = None,
    ) -> EvalResult:
        """
        Run the benchmark on a model.

        Args:
            model_path: Path to the model or model name.
            num_samples: Optional limit on samples to evaluate.

        Returns:
            EvalResult with scores and metrics.
        """
        ...


class CustomEvalBenchmark(Benchmark):
    """
    Custom evaluation benchmark from a JSONL file.

    Each line should have:
    - "input": The input prompt
    - "expected": The expected output or acceptable outputs
    - "category": Optional category for breakdown
    """

    def __init__(self, eval_file: str, name: str | None = None):
        self.eval_file = Path(eval_file)
        self._name = name or self.eval_file.stem

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Custom evaluation from {self.eval_file.name}"

    async def evaluate(
        self,
        model_path: str,
        num_samples: int | None = None,
    ) -> EvalResult:
        """Run custom evaluation."""
        samples = []
        with open(self.eval_file) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
                    if num_samples and len(samples) >= num_samples:
                        break

        # This would integrate with the model for inference
        # For now, return a placeholder structure
        return EvalResult(
            benchmark_name=self.name,
            model_name=model_path,
            score=0.0,
            max_score=len(samples),
            samples_evaluated=len(samples),
            metrics={
                "accuracy": 0.0,
                "samples": len(samples),
            },
        )


class AccuracyBenchmark(Benchmark):
    """
    Simple accuracy benchmark for classification/QA tasks.

    Compares model outputs to expected answers.
    """

    def __init__(
        self,
        test_data: list[dict[str, Any]],
        name: str = "accuracy",
    ):
        self.test_data = test_data
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Accuracy evaluation on test data"

    async def evaluate(
        self,
        model_path: str,
        num_samples: int | None = None,
    ) -> EvalResult:
        samples = self.test_data
        if num_samples:
            samples = samples[:num_samples]

        # Placeholder - would run inference
        return EvalResult(
            benchmark_name=self.name,
            model_name=model_path,
            score=0.0,
            max_score=len(samples),
            samples_evaluated=len(samples),
        )


class PerplexityBenchmark(Benchmark):
    """Measure perplexity on a test set."""

    def __init__(self, test_file: str):
        self.test_file = Path(test_file)

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def description(self) -> str:
        return "Perplexity evaluation on held-out data"

    async def evaluate(
        self,
        model_path: str,
        num_samples: int | None = None,
    ) -> EvalResult:
        # Would compute actual perplexity
        return EvalResult(
            benchmark_name=self.name,
            model_name=model_path,
            score=0.0,  # Lower is better for perplexity
            max_score=0.0,
            metrics={"perplexity": 0.0},
        )


class EvalSuite:
    """
    Collection of benchmarks for comprehensive evaluation.
    """

    def __init__(self):
        self.benchmarks: list[Benchmark] = []

    def add(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)

    def add_custom(self, eval_file: str, name: str | None = None) -> None:
        """Add a custom evaluation from a file."""
        self.add(CustomEvalBenchmark(eval_file, name))

    async def run_all(
        self,
        model_path: str,
        num_samples: int | None = None,
    ) -> list[EvalResult]:
        """Run all benchmarks on a model."""
        results = []
        for benchmark in self.benchmarks:
            result = await benchmark.evaluate(model_path, num_samples)
            results.append(result)
        return results

    async def compare_models(
        self,
        model_paths: list[str],
        num_samples: int | None = None,
    ) -> dict[str, list[EvalResult]]:
        """Compare multiple models across all benchmarks."""
        comparisons = {}
        for model_path in model_paths:
            results = await self.run_all(model_path, num_samples)
            comparisons[model_path] = results
        return comparisons


def format_eval_results(results: list[EvalResult]) -> str:
    """Format evaluation results as a table."""
    lines = ["Evaluation Results", "=" * 60]

    for result in results:
        lines.append(f"\n{result.benchmark_name}")
        lines.append("-" * 40)
        lines.append(f"Model: {result.model_name}")
        lines.append(f"Score: {result.score:.2f} / {result.max_score:.2f}")
        lines.append(f"Samples: {result.samples_evaluated} (errors: {result.errors})")

        if result.metrics:
            lines.append("Metrics:")
            for key, value in result.metrics.items():
                lines.append(f"  {key}: {value:.4f}")

    return "\n".join(lines)
