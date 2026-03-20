"""
ONNX Optimization Module
Model optimization using ONNX runtime for faster inference.
"""

import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from ..core import ClassificationResult, get_logger

logger = get_logger(__name__)


class ONNXClassifier:
    """
    ONNX Runtime optimized classifier for faster inference.

    Provides 2-5x speedup compared to PyTorch for CPU inference.
    """

    def __init__(
        self,
        onnx_model_path: str | Path,
        use_cuda: bool = False,
        num_threads: int = 4,
    ):
        """
        Initialize ONNX classifier.

        Args:
            onnx_model_path: Path to ONNX model file
            use_cuda: Use GPU acceleration
            num_threads: Number of CPU threads
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime-gpu is required. Install with: pip install onnxruntime-gpu"
            )

        self.model_path = Path(onnx_model_path)
        self.use_cuda = use_cuda

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Configure session options
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = num_threads
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Select execution provider
        if use_cuda and ort.get_device() == "GPU":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Using CUDA execution provider")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info("Using CPU execution provider")

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers,
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ONNX model loaded: {self.model_path}")

    def predict(self, input_tensor: np.ndarray) -> dict[str, float]:
        """
        Run inference on input tensor.

        Args:
            input_tensor: Input tensor [1, 3, 299, 299]

        Returns:
            Dictionary with real, fake, confidence scores
        """
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor.astype(np.float32)},
        )

        # Apply softmax
        logits = outputs[0][0]
        probs = self._softmax(logits)

        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        return {
            "real": real_prob,
            "fake": fake_prob,
            "confidence": max(real_prob, fake_prob),
        }

    def classify(self, input_tensor: np.ndarray) -> ClassificationResult:
        """
        Classify input as real or fake.

        Args:
            input_tensor: Input tensor [1, 3, 299, 299]

        Returns:
            ClassificationResult
        """
        result = self.predict(input_tensor)

        prediction = "FAKE" if result["fake"] > result["real"] else "REAL"

        return ClassificationResult(
            prediction=prediction,
            real_probability=result["real"],
            fake_probability=result["fake"],
            confidence=result["confidence"],
        )

    def predict_batch(self, input_tensors: np.ndarray) -> list[dict[str, float]]:
        """
        Run batch inference.

        Args:
            input_tensors: Batch of input tensors [N, 3, 299, 299]

        Returns:
            List of prediction dictionaries
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensors.astype(np.float32)},
        )

        logits = outputs[0]
        results = []

        for logit in logits:
            probs = self._softmax(logit)
            real_prob = float(probs[0])
            fake_prob = float(probs[1])

            results.append(
                {
                    "real": real_prob,
                    "fake": fake_prob,
                    "confidence": max(real_prob, fake_prob),
                }
            )

        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def get_benchmark_info(self) -> dict:
        """Get benchmark information about the model."""
        return {
            "model_path": str(self.model_path),
            "use_cuda": self.use_cuda,
            "execution_providers": self.session.get_providers(),
            "input_name": self.input_name,
            "output_name": self.output_name,
        }


def export_pytorch_to_onnx(
    pytorch_model,
    output_path: str | Path,
    input_shape: tuple = (1, 3, 299, 299),
    opset_version: int = 14,
) -> Path:
    """
    Export PyTorch model to ONNX format.

    Args:
        pytorch_model: PyTorch model to export
        output_path: Output path for ONNX model
        input_shape: Input tensor shape
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export
    pytorch_model.eval()
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"Model exported to ONNX: {output_path}")

    # Verify export
    if ONNX_AVAILABLE:
        onnx_model = ort.InferenceSession(str(output_path))
        logger.info("ONNX model verified successfully")

    return output_path
