"""
Deepfake Classifier Module
XceptionNet-based deepfake detection.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Optional
from torchvision import transforms
import logging

from ..core import (
    ClassificationResult, 
    ClassifierConfig, 
    CLASSIFIER_CONFIG,
    YOINK_DIR,
    DEFAULT_WEIGHTS_PATH
)
from ..core.exceptions import ModelNotFoundError, ModelLoadError

# Add yoink path for model imports
sys.path.insert(0, str(YOINK_DIR))
from network.models import model_selection

logger = logging.getLogger(__name__)


class DeepfakeClassifier:
    """
    Classifies face images as real or fake using XceptionNet.
    """
    
    def __init__(
        self, 
        weights_path: Optional[str] = None,
        use_cuda: bool = False,
        config: ClassifierConfig = None
    ):
        """
        Initialize the deepfake classifier.
        
        Args:
            weights_path: Path to pretrained weights
            use_cuda: Whether to use GPU
            config: Classifier configuration
        """
        self.config = config or CLASSIFIER_CONFIG
        self.weights_path = weights_path or str(DEFAULT_WEIGHTS_PATH)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        self.transform = transforms.Compose([
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                list(self.config.normalize_mean), 
                list(self.config.normalize_std)
            )
        ])
        
        self.softmax = nn.Softmax(dim=1)
        self.model = self._load_model()
        
        logger.info(f"Classifier initialized on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load the XceptionNet model with pretrained weights."""
        logger.info("Loading XceptionNet model...")
        
        model = model_selection(
            modelname=self.config.model_name, 
            num_out_classes=self.config.num_classes,
            dropout=self.config.dropout
        )
        
        import os
        if os.path.exists(self.weights_path):
            logger.info(f"Loading weights from {self.weights_path}")
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info("Weights loaded successfully!")
            except Exception as e:
                raise ModelLoadError(f"Failed to load model weights: {e}")
        else:
            raise ModelNotFoundError(f"Weights not found at {self.weights_path}")
        
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess an image for model input."""
        import cv2
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def classify(self, face_image: np.ndarray) -> ClassificationResult:
        """
        Classify a face image as real or fake.
        
        Args:
            face_image: BGR face image
            
        Returns:
            ClassificationResult
        """
        input_tensor = self.preprocess(face_image)
        output = self.model(input_tensor)
        probs = self.softmax(output).cpu().numpy()[0]
        
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        prediction = "FAKE" if fake_prob > real_prob else "REAL"
        
        return ClassificationResult(
            prediction=prediction,
            real_probability=real_prob,
            fake_probability=fake_prob,
            confidence=max(real_prob, fake_prob)
        )
    
    @torch.no_grad()
    def classify_batch(self, face_images: List[np.ndarray]) -> List[ClassificationResult]:
        """Classify multiple face images in a single batch."""
        if not face_images:
            return []
        
        tensors = [self.preprocess(img) for img in face_images]
        batch = torch.cat(tensors, dim=0)
        
        outputs = self.model(batch)
        probs = self.softmax(outputs).cpu().numpy()
        
        results = []
        for p in probs:
            real_prob, fake_prob = float(p[0]), float(p[1])
            results.append(ClassificationResult(
                prediction="FAKE" if fake_prob > real_prob else "REAL",
                real_probability=real_prob,
                fake_probability=fake_prob,
                confidence=max(real_prob, fake_prob)
            ))
        
        return results
