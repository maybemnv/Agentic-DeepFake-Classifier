"""
Dataset Benchmark Report
Accuracy/speed comparison scaffold for standard deepfake detection datasets.

Usage:
    python -m benchmarks.dataset_report

To populate with real numbers, run your detector on each dataset and call
DatasetBenchmarks.add_result() before printing the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent


@dataclass
class DatasetResult:
    """Accuracy and throughput metrics for one model on one dataset."""

    dataset: str
    model: str
    accuracy: float | None = None       # 0-1
    auc: float | None = None            # 0-1
    avg_inference_ms: float | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Reference numbers from published papers (as of their respective release).
# Replace with your own eval results when available.
# ---------------------------------------------------------------------------
PAPER_BASELINES: list[DatasetResult] = [
    DatasetResult(
        dataset="FaceForensics++ (c23)",
        model="XceptionNet (Rossler et al. 2019)",
        accuracy=0.9954,
        auc=0.9995,
        avg_inference_ms=None,
        notes="Paper: https://arxiv.org/abs/1901.08971",
    ),
    DatasetResult(
        dataset="FaceForensics++ (c40)",
        model="XceptionNet (Rossler et al. 2019)",
        accuracy=0.9554,
        auc=0.9862,
        avg_inference_ms=None,
        notes="Paper: https://arxiv.org/abs/1901.08971",
    ),
    DatasetResult(
        dataset="DFDC (Dolhansky et al. 2020)",
        model="XceptionNet (baseline)",
        accuracy=0.6532,
        auc=0.7212,
        avg_inference_ms=None,
        notes="Paper: https://arxiv.org/abs/2006.07397",
    ),
    DatasetResult(
        dataset="Celeb-DF v2 (Li et al. 2020)",
        model="XceptionNet (baseline)",
        accuracy=None,
        auc=0.7350,
        avg_inference_ms=None,
        notes="Paper: https://arxiv.org/abs/1909.12962",
    ),
    # -----------------------------------------------------------------------
    # YOUR RESULTS — fill these in after running evaluate.py on each dataset
    # -----------------------------------------------------------------------
    DatasetResult(
        dataset="FaceForensics++ (c23)",
        model="This project (Agentic DeepFake Classifier)",
        accuracy=None,
        auc=None,
        avg_inference_ms=None,
        notes="TODO: run benchmarks/run.py against FF++ c23 split",
    ),
    DatasetResult(
        dataset="DFDC",
        model="This project (Agentic DeepFake Classifier)",
        accuracy=None,
        auc=None,
        avg_inference_ms=None,
        notes="TODO: run benchmarks/run.py against DFDC test split",
    ),
]


class DatasetBenchmarks:
    """Collect and display dataset-level benchmark results."""

    def __init__(self) -> None:
        self._results: list[DatasetResult] = list(PAPER_BASELINES)

    def add_result(self, result: DatasetResult) -> None:
        """Append a new measured result."""
        self._results.append(result)

    def print_comparison_table(self) -> None:
        """Print a formatted comparison table to stdout."""
        header = f"{'Dataset':<35} {'Model':<45} {'Acc':>6} {'AUC':>6} {'ms/frame':>10} Notes"
        sep = "-" * len(header)
        print("\n" + sep)
        print("DATASET ACCURACY / SPEED COMPARISON")
        print(sep)
        print(header)
        print(sep)
        for r in self._results:
            acc = f"{r.accuracy:.3f}" if r.accuracy is not None else "  N/A"
            auc = f"{r.auc:.3f}" if r.auc is not None else "  N/A"
            ms = f"{r.avg_inference_ms:.1f}" if r.avg_inference_ms is not None else "      N/A"
            model = r.model[:43]
            print(f"{r.dataset:<35} {model:<45} {acc:>6} {auc:>6} {ms:>10}  {r.notes}")
        print(sep)
        print(dedent("""
        Legend:
          Acc  — frame-level binary accuracy on the test split
          AUC  — area under ROC curve
          N/A  — not yet measured; update DatasetResult fields and re-run

        To populate your own numbers:
          1. Download the dataset (see dataset homepage for access forms).
          2. Run: python benchmarks/run.py --dataset <path> --output benchmark_results/
          3. Fill in the 'This project' rows above with the printed metrics.
        """))


if __name__ == "__main__":
    DatasetBenchmarks().print_comparison_table()
