"""
Deepfake Classifier Module
XceptionNet-based deepfake detection using FaceForensics++ pretrained weights.

Author: Agentic Deepfake Classifier
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

# Add yoink path for model imports
YOINK_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'yoink', 'Deepfake-Detection')
sys.path.insert(0, os.path.abspath(YOINK_PATH))

from network.models import model_selection
from torchvision import transforms

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of deepfake classification."""
    prediction: str  # "REAL" or "FAKE"
    real_probability: float
    fake_probability: float
    confidence: float  # Max probability
    
    @property
    def is_fake(self) -> bool:
        return self.prediction == "FAKE"


class DeepfakeClassifier:
    """
    Classifies face images as real or fake using XceptionNet.
    
    Uses pretrained weights from FaceForensics++ dataset.
    Runs in CPU mode for edge deployment compatibility.
    """
    
    DEFAULT_WEIGHTS_PATH = os.path.join(
        os.path.dirname(__file__), '..', '..', 'model', 'ffpp_c23.pth'
    )
    
    def __init__(
        self, 
        weights_path: Optional[str] = None,
        use_cuda: bool = False,
        dropout: float = 0.5
    ):
        """
        Initialize the deepfake classifier.
        
        Args:
            weights_path: Path to pretrained weights (.pth file)
            use_cuda: Whether to use GPU (default: False for edge deployment)
            dropout: Dropout rate for model (default: 0.5)
        """
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS_PATH
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # Initialize transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Softmax for probability output
        self.softmax = nn.Softmax(dim=1)
        
        # Load model
        self.model = self._load_model(dropout)
        logger.info(f"Classifier initialized on {self.device}")
    
    def _load_model(self, dropout: float) -> nn.Module:
        """
        Load the XceptionNet model with pretrained weights.
        
        Args:
            dropout: Dropout rate
            
        Returns:
            Loaded PyTorch model in eval mode
        """
        logger.info("Loading XceptionNet model...")
        
        # Create model architecture
        model = model_selection(
            modelname='xception', 
            num_out_classes=2,
            dropout=dropout
        )
        
        # Load pretrained weights
        if os.path.exists(self.weights_path):
            logger.info(f"Loading weights from {self.weights_path}")
            state_dict = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info("Weights loaded successfully!")
        else:
            logger.warning(f"Weights not found at {self.weights_path}!")
            logger.warning("Model will run with random weights (inaccurate predictions)")
        
        # Handle DataParallel if needed
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: BGR image (OpenCV format) or RGB numpy array
            
        Returns:
            Preprocessed tensor ready for model input
        """
        import cv2
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def classify(self, face_image: np.ndarray) -> ClassificationResult:
        """
        Classify a face image as real or fake.
        
        Args:
            face_image: BGR face image (OpenCV format)
            
        Returns:
            ClassificationResult with prediction and probabilities
        """
        # Preprocess
        input_tensor = self.preprocess(face_image)
        
        # Run inference
        output = self.model(input_tensor)
        
        # Get probabilities
        probs = self.softmax(output)
        probs = probs.cpu().numpy()[0]
        
        # Index 0 = real, Index 1 = fake (based on FaceForensics++ convention)
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        
        # Make prediction
        prediction = "FAKE" if fake_prob > real_prob else "REAL"
        confidence = max(real_prob, fake_prob)
        
        return ClassificationResult(
            prediction=prediction,
            real_probability=real_prob,
            fake_probability=fake_prob,
            confidence=confidence
        )
    
    @torch.no_grad()
    def classify_batch(
        self, 
        face_images: list
    ) -> list:
        """
        Classify multiple face images in a single batch.
        
        Args:
            face_images: List of BGR face images
            
        Returns:
            List of ClassificationResult objects
        """
        if not face_images:
            return []
        
        # Preprocess all images
        tensors = [self.preprocess(img) for img in face_images]
        batch = torch.cat(tensors, dim=0)
        
        # Run inference
        outputs = self.model(batch)
        probs = self.softmax(outputs)
        probs = probs.cpu().numpy()
        
        # Convert to results
        results = []
        for p in probs:
            real_prob = float(p[0])
            fake_prob = float(p[1])
            prediction = "FAKE" if fake_prob > real_prob else "REAL"
            
            results.append(ClassificationResult(
                prediction=prediction,
                real_probability=real_prob,
                fake_probability=fake_prob,
                confidence=max(real_prob, fake_prob)
            ))
        
        return results
