"""
Deepfake Classifier Module
Pure inference class for XceptionNet-based deepfake classification using Keras.
Only tensors in, probabilities out. No video logic, no face detection, no file IO.
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from torchvision import transforms
import logging
from huggingface_hub import hf_hub_download

from ..core import (
    ClassificationResult,
    ClassifierConfig,
    CLASSIFIER_CONFIG,
)
from ..core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class DeepfakeClassifier:
    """
    Pure inference class for XceptionNet deepfake classification using Keras 3.

    ONLY accepts pre-processed face tensors (BGR numpy arrays subject to internal transform).
    ONLY returns probability distributions (real, fake).
    """

    HF_REPO_ID = "Redgerd/XceptionNet-Keras"
    HF_FILENAME = "XceptionNet.keras"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_cuda: bool = False,
        config: ClassifierConfig = None,
    ):
        self.config = config or CLASSIFIER_CONFIG
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    list(self.config.normalize_mean), list(self.config.normalize_std)
                ),
            ]
        )

        self.model = self._load_model()
        logger.info(f"DeepfakeClassifier initialized on {self.device}")

    def _load_model(self):
        try:
            logger.info(f"Loading model from Hugging Face: {self.HF_REPO_ID}")

            local_path = hf_hub_download(
                repo_id=self.HF_REPO_ID, filename=self.HF_FILENAME, repo_type="model"
            )

            logger.info(f"Loading Keras model from: {local_path}")
            model = keras.saving.load_model(local_path)

            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load Keras model: {e}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a face image for model input.

        Args:
            image: BGR numpy array (face cropped, any size)

        Returns:
            Preprocessed tensor ready for inference
        """
        import cv2

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)
        # Transform returns a torch Tensor [3, 299, 299]
        tensor = self.transform(pil_image)

        # Add batch dimension [1, 3, 299, 299]
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def classify(self, face_image: np.ndarray) -> ClassificationResult:
        """
        Classify a face image as real or fake.

        Args:
            face_image: BGR numpy array of cropped face

        Returns:
            ClassificationResult
        """
        input_tensor = self.preprocess(face_image)

        # Keras 3 model call with Torch backend accepts Torch tensors
        # Output is expected to be [1, 2] logits or probs depending on model
        # The Redgerd/XceptionNet-Keras model likely returns logits.
        # Let's assume standard behavior and apply softmax if needed,
        # but pure Keras models often include activation if it's "inference ready".
        # However, safe to check.

        # NOTE: Keras models usually expect channel-last [N, H, W, C] unless configured otherwise.
        # With torch backend, it follows image_data_format(). Default is usually 'channels_last'.
        # We might need to permute.

        if keras.config.image_data_format() == "channels_last":
            # Input tensor is [1, 3, 299, 299] (channels_first)
            input_tensor = input_tensor.permute(0, 2, 3, 1)  # [1, 299, 299, 3]

        output = self.model(input_tensor)

        # Convert output to numpy
        if hasattr(output, "detach"):
            probs = output.detach().cpu().numpy()[0]
        else:
            probs = np.array(output)[0]

        # Ensure probabilities sum to 1 (apply softmax if raw logits)
        # Deepfake models often output raw logits.
        # If values are not in [0, 1], apply softmax.
        if np.any(probs < 0) or np.any(probs > 1.01):
            exp_preds = np.exp(probs)
            probs = exp_preds / np.sum(exp_preds)

        real_prob = float(probs[0])
        # Assumption: Index 0 is Real, Index 1 is Fake.
        # Verify this with model card or test. Standard FF++ is usually 0: Fake, 1: Real OR 0: Real, 1: Fake.
        # Let's stick to existing logic: 0=Real, 1=Fake.
        fake_prob = float(probs[1])

        prediction = "FAKE" if fake_prob > real_prob else "REAL"

        return ClassificationResult(
            prediction=prediction,
            real_probability=real_prob,
            fake_probability=fake_prob,
            confidence=max(real_prob, fake_prob),
        )

    def classify_batch(
        self, face_images: List[np.ndarray]
    ) -> List[ClassificationResult]:
        if not face_images:
            return []

        # Preprocess all
        tensors = []
        for img in face_images:
            t = self.preprocess(img)
            tensors.append(t)

        batch = torch.cat(tensors, dim=0)  # [N, 3, H, W]

        if keras.config.image_data_format() == "channels_last":
            batch = batch.permute(0, 2, 3, 1)  # [N, H, W, 3]

        outputs = self.model(batch)

        if hasattr(outputs, "detach"):
            probs_batch = outputs.detach().cpu().numpy()
        else:
            probs_batch = np.array(outputs)

        results = []
        for probs in probs_batch:
            if np.any(probs < 0) or np.any(probs > 1.01):
                exp_preds = np.exp(probs)
                probs = exp_preds / np.sum(exp_preds)

            real_prob, fake_prob = float(probs[0]), float(probs[1])
            results.append(
                ClassificationResult(
                    prediction="FAKE" if fake_prob > real_prob else "REAL",
                    real_probability=real_prob,
                    fake_probability=fake_prob,
                    confidence=max(real_prob, fake_prob),
                )
            )
        return results
