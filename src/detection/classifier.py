"""
Deepfake Classifier Module
Pure PyTorch inference class for XceptionNet-based deepfake classification.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any
from torchvision import transforms
import logging

from ..core import (
    ClassificationResult,
    ClassifierConfig,
    CLASSIFIER_CONFIG,
    PROJECT_ROOT
)
from ..core.exceptions import ModelLoadError
from .network import model_selection

logger = logging.getLogger(__name__)


class DeepfakeClassifier:
    """
    Pure inference class for XceptionNet deepfake classification using PyTorch.

    Model: FaceForensics++ XceptionNet (Transfer Learning)
    Architecture: Xception
    Weights: ffpp_c23.pth (Compressed C23)
    """

    MODEL_RELATIVE_PATH = "model/ffpp_c23.pth"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_cuda: bool = False,
        config: ClassifierConfig = None,
    ):
        """
        Initialize the PyTorch Deepfake Classifier.

        Args:
            weights_path: Path to the .pth weights file. If None, uses default project path.
            use_cuda: Whether to use CUDA if available.
            config: Classifier configuration.
        """
        self.config = config or CLASSIFIER_CONFIG
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Standard Xception/ImageNet preprocessing
        # Resize to 299x299, Normalize mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    list(self.config.normalize_mean), list(self.config.normalize_std)
                ),
            ]
        )

        # Resolve weights path
        if not weights_path:
            self.weights_path = os.path.join(PROJECT_ROOT, self.MODEL_RELATIVE_PATH)
        else:
            self.weights_path = weights_path

        self.model = self._load_model()
        logger.info(f"DeepfakeClassifier initialized on {self.device}")

    def _load_model(self) -> nn.Module:
        """
        Load the PyTorch model and weights.
        """
        try:
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"Model weights not found at: {self.weights_path}")

            logger.info(f"Loading PyTorch model from: {self.weights_path}")

            # Instantiate the model structure
            # FaceForensics++ models typically use xception with 2 classes
            model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)

            # Load state dict
            # map_location ensures we can load GPU weights on CPU if needed
            state_dict = torch.load(self.weights_path, map_location=self.device)
            
            # Handle potential DataParallel wrapping and compatibility with pretrained weights
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]
                
                # Fix for FaceForensics++ pretrained weights compatibility
                if 'last_linear' in name:
                    name = name.replace('last_linear', 'fc')
                    
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)

            # Set to eval mode and move to device
            model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Failed to load PyTorch model: {e}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a face image for model input.

        Args:
            image: BGR numpy array (face cropped, any size)

        Returns:
            Preprocessed tensor [1, 3, 299, 299]
        """
        import cv2

        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)
        # Transform returns a torch Tensor [3, 299, 299]
        tensor = self.transform(pil_image)

        # Add batch dimension [1, 3, 299, 299]
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, face_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Perform inference on a preprocessed face tensor.

        Args:
            face_tensor: Torch tensor of shape [1, 3, 299, 299]

        Returns:
            Dictionary with 'real', 'fake', 'confidence' scores.
        """
        with torch.no_grad():
            logits = self.model(face_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        
        return {
            "real": real_prob,
            "fake": fake_prob,
            "confidence": max(real_prob, fake_prob)
        }

    def classify(self, face_image: np.ndarray) -> ClassificationResult:
        """
        Classify a face image as real or fake.

        Args:
            face_image: BGR numpy array of cropped face

        Returns:
            ClassificationResult
        """
        input_tensor = self.preprocess(face_image)
        result = self.predict(input_tensor)

        prediction = "FAKE" if result["fake"] > result["real"] else "REAL"

        return ClassificationResult(
            prediction=prediction,
            real_probability=result["real"],
            fake_probability=result["fake"],
            confidence=result["confidence"],
        )

    def classify_batch(
        self, face_images: List[np.ndarray]
    ) -> List[ClassificationResult]:
        """
        Classify a batch of faces.
        """
        if not face_images:
            return []

        # Preprocess all
        tensors = []
        for img in face_images:
            t = self.preprocess(img)
            tensors.append(t)

        # Concatenate: [N, 3, 299, 299]
        batch_tensor = torch.cat(tensors, dim=0)

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs_batch = F.softmax(logits, dim=1).cpu().numpy()

        results = []
        for probs in probs_batch:
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
