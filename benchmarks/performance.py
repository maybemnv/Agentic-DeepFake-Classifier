"""
Performance Benchmarks
Benchmark tools for measuring model performance.
"""

import time
import numpy as np
from typing import Callable
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time_seconds: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    throughput_per_second: float


def benchmark_function(
    func: Callable,
    *args,
    iterations: int = 100,
    warmup: int = 10,
    name: str | None = None,
    **kwargs,
) -> BenchmarkResult:
    """
    Benchmark a function's execution time.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to function
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        name: Benchmark name
        **kwargs: Keyword arguments to pass to function

    Returns:
        BenchmarkResult with timing statistics
    """
    benchmark_name = name or func.__name__

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    total_time = sum(times) / 1000  # Convert to seconds
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    throughput = iterations / total_time if total_time > 0 else 0

    result = BenchmarkResult(
        name=benchmark_name,
        iterations=iterations,
        total_time_seconds=total_time,
        avg_time_ms=avg_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        std_time_ms=std_time,
        throughput_per_second=throughput,
    )

    logger.info(f"Benchmark: {benchmark_name}")
    logger.info(f"  Iterations: {iterations}")
    logger.info(f"  Avg time: {avg_time:.2f}ms")
    logger.info(f"  Min time: {min_time:.2f}ms")
    logger.info(f"  Max time: {max_time:.2f}ms")
    logger.info(f"  Std dev: {std_time:.2f}ms")
    logger.info(f"  Throughput: {throughput:.2f} ops/sec")

    return result


class ModelBenchmarks:
    """Benchmarks for deepfake detection model."""

    def __init__(self, classifier):
        """
        Initialize model benchmarks.

        Args:
            classifier: DeepfakeClassifier instance
        """
        self.classifier = classifier
        self.results: dict[str, BenchmarkResult] = {}

    def benchmark_inference(self, face_tensor, iterations: int = 100) -> BenchmarkResult:
        """
        Benchmark model inference time.

        Args:
            face_tensor: Preprocessed face tensor
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        result = benchmark_function(
            self.classifier.predict,
            face_tensor,
            iterations=iterations,
            name="model_inference",
        )
        self.results["inference"] = result
        return result

    def benchmark_preprocessing(self, face_image: np.ndarray, iterations: int = 100) -> BenchmarkResult:
        """
        Benchmark preprocessing time.

        Args:
            face_image: Face image numpy array
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        result = benchmark_function(
            self.classifier.preprocess,
            face_image,
            iterations=iterations,
            name="preprocessing",
        )
        self.results["preprocessing"] = result
        return result

    def benchmark_full_pipeline(self, face_image: np.ndarray, iterations: int = 100) -> BenchmarkResult:
        """
        Benchmark full classification pipeline.

        Args:
            face_image: Face image numpy array
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        result = benchmark_function(
            self.classifier.classify,
            face_image,
            iterations=iterations,
            name="full_pipeline",
        )
        self.results["full_pipeline"] = result
        return result

    def benchmark_batch_inference(
        self,
        face_images: list[np.ndarray],
        iterations: int = 50,
    ) -> BenchmarkResult:
        """
        Benchmark batch inference.

        Args:
            face_images: List of face images
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        result = benchmark_function(
            self.classifier.classify_batch,
            face_images,
            iterations=iterations,
            name="batch_inference",
        )
        self.results["batch_inference"] = result
        return result

    def save_results(self, output_path: str | Path):
        """
        Save benchmark results to JSON file.

        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        results_dict = {
            name: {
                "name": result.name,
                "iterations": result.iterations,
                "total_time_seconds": result.total_time_seconds,
                "avg_time_ms": result.avg_time_ms,
                "min_time_ms": result.min_time_ms,
                "max_time_ms": result.max_time_ms,
                "std_time_ms": result.std_time_ms,
                "throughput_per_second": result.throughput_per_second,
            }
            for name, result in self.results.items()
        }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Benchmark results saved to: {output_path}")

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for name, result in self.results.items():
            print(f"\n{name.upper()}:")
            print(f"  Avg: {result.avg_time_ms:.2f}ms | Throughput: {result.throughput_per_second:.1f} ops/sec")

        print("=" * 60)


class VideoProcessingBenchmarks:
    """Benchmarks for video processing pipeline."""

    def __init__(self, analyzer):
        """
        Initialize video processing benchmarks.

        Args:
            analyzer: DeepfakeAnalyzer instance
        """
        self.analyzer = analyzer
        self.results: dict[str, BenchmarkResult] = {}

    def benchmark_video_analysis(self, video_path: str, iterations: int = 10) -> BenchmarkResult:
        """
        Benchmark full video analysis.

        Args:
            video_path: Path to video file
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        result = benchmark_function(
            self.analyzer.analyze,
            video_path,
            iterations=iterations,
            name="video_analysis",
            show_progress=False,
        )
        self.results["video_analysis"] = result
        return result

    def benchmark_quick_check(self, video_path: str, iterations: int = 20) -> BenchmarkResult:
        """
        Benchmark quick check mode.

        Args:
            video_path: Path to video file
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        result = benchmark_function(
            self.analyzer.quick_check,
            video_path,
            iterations=iterations,
            name="quick_check",
        )
        self.results["quick_check"] = result
        return result


def run_all_benchmarks(classifier, analyzer, test_video: str, output_dir: str = "benchmark_results"):
    """
    Run all benchmarks and save results.

    Args:
        classifier: DeepfakeClassifier instance
        analyzer: DeepfakeAnalyzer instance
        test_video: Path to test video
        output_dir: Directory to save results
    """
    import cv2
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create test face image
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (200, 100), (440, 400), (200, 200, 200), -1)

    # Model benchmarks
    print("\n🔍 Running Model Benchmarks...")
    model_benchmarks = ModelBenchmarks(classifier)
    model_benchmarks.benchmark_preprocessing(test_frame)
    model_benchmarks.benchmark_full_pipeline(test_frame)
    model_benchmarks.save_results(output_path / "model_benchmarks.json")
    model_benchmarks.print_summary()

    # Video benchmarks
    print("\n🔍 Running Video Processing Benchmarks...")
    video_benchmarks = VideoProcessingBenchmarks(analyzer)
    video_benchmarks.benchmark_video_analysis(test_video, iterations=5)
    video_benchmarks.benchmark_quick_check(test_video, iterations=10)
    video_benchmarks.save_results(output_path / "video_benchmarks.json")
    video_benchmarks.print_summary()

    # Overall summary
    print("\n✅ All benchmarks completed!")
    print(f"Results saved to: {output_path}")

    return {
        "model": model_benchmarks.results,
        "video": video_benchmarks.results,
    }
