"""
Benchmark Runner
CLI script to run performance benchmarks.
"""

import argparse
import sys
from pathlib import Path

import torch
import cv2
import numpy as np

from src.detection import DeepfakeClassifier, ONNXClassifier, DeepfakeAnalyzer
from benchmarks.performance import (
    ModelBenchmarks,
    VideoProcessingBenchmarks,
    run_all_benchmarks,
)


def create_test_face() -> np.ndarray:
    """Create a synthetic test face image."""
    # Create gray gradient image
    face = np.zeros((299, 299, 3), dtype=np.uint8)
    for i in range(299):
        face[i, :] = [i, i, i]
    return face


def benchmark_pytorch(use_cuda: bool = False, iterations: int = 100):
    """Benchmark PyTorch classifier."""
    print("\n" + "=" * 60)
    print("PyTorch Classifier Benchmarks")
    print("=" * 60)

    classifier = DeepfakeClassifier(use_cuda=use_cuda)
    test_face = create_test_face()

    benchmarks = ModelBenchmarks(classifier)
    benchmarks.benchmark_preprocessing(test_face, iterations=iterations)
    benchmarks.benchmark_full_pipeline(test_face, iterations=iterations)
    benchmarks.print_summary()

    return benchmarks


def benchmark_onnx(use_cuda: bool = False, iterations: int = 100):
    """Benchmark ONNX classifier."""
    print("\n" + "=" * 60)
    print("ONNX Classifier Benchmarks")
    print("=" * 60)

    try:
        classifier = ONNXClassifier("model/ffpp_c23.onnx", use_cuda=use_cuda)
        test_face = create_test_face()

        # Preprocess for ONNX
        pil_image = classifier.classifier.transform(
            Image.fromarray(cv2.cvtColor(test_face, cv2.COLOR_BGR2RGB))
        )
        tensor = pil_image.unsqueeze(0).numpy()

        benchmarks = ModelBenchmarks(classifier)
        benchmarks.benchmark_inference(tensor, iterations=iterations)
        benchmarks.print_summary()

        return benchmarks
    except FileNotFoundError:
        print("ONNX model not found. Export PyTorch model first.")
        return None
    except ImportError:
        print("ONNX Runtime not installed. Install with: pip install onnxruntime-gpu")
        return None


def benchmark_video(video_path: str, use_cuda: bool = False, iterations: int = 10):
    """Benchmark video analysis."""
    print("\n" + "=" * 60)
    print("Video Analysis Benchmarks")
    print("=" * 60)

    classifier = DeepfakeClassifier(use_cuda=use_cuda)
    analyzer = DeepfakeAnalyzer(classifier, sample_rate=1.0, max_frames=50)

    benchmarks = VideoProcessingBenchmarks(analyzer)
    benchmarks.benchmark_video_analysis(video_path, iterations=iterations)
    benchmarks.benchmark_quick_check(video_path, iterations=iterations * 2)
    benchmarks.print_summary()

    return benchmarks


def export_onnx_model():
    """Export PyTorch model to ONNX."""
    print("\nExporting PyTorch model to ONNX...")

    classifier = DeepfakeClassifier(use_cuda=False)
    model = classifier.model

    output_path = Path("model/ffpp_c23.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.detection.onnx_classifier import export_pytorch_to_onnx

    export_pytorch_to_onnx(model, output_path)
    print(f"Model exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks for Agentic DeepFake Classifier"
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA for benchmarks",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmarks",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video for video analysis benchmarks",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export PyTorch model to ONNX",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )

    args = parser.parse_args()

    if args.export_onnx:
        export_onnx_model()
        return

    if args.all or (args.video and not args.all):
        if args.video is None:
            print("Error: --video required for video benchmarks")
            sys.exit(1)

        if not Path(args.video).exists():
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)

        classifier = DeepfakeClassifier(use_cuda=args.cuda)
        analyzer = DeepfakeAnalyzer(classifier)

        run_all_benchmarks(
            classifier,
            analyzer,
            test_video=args.video,
            output_dir=args.output,
        )
    else:
        # Run model benchmarks
        benchmark_pytorch(use_cuda=args.cuda, iterations=args.iterations)

        # Try ONNX if available
        onnx_path = Path("model/ffpp_c23.onnx")
        if onnx_path.exists():
            benchmark_onnx(use_cuda=args.cuda, iterations=args.iterations)
        else:
            print("\nSkip ONNX benchmarks (model not found)")
            print("Run with --export-onnx to create ONNX model")

    print("\n" + "=" * 60)
    print("Benchmarks completed!")
    print("=" * 60)


if __name__ == "__main__":
    from PIL import Image

    main()
