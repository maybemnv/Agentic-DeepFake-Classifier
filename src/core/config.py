"""
Configuration Module
Central place for all configuration values and magic numbers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "model"



# =============================================================================
# VIDEO PROCESSING
# =============================================================================

@dataclass(frozen=True)
class VideoConfig:
    """Video processing configuration."""
    sample_rate: float = 1.0  # Frames per second to extract
    max_frames: int = None    # None = no limit
    supported_formats: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


# =============================================================================
# FACE DETECTION
# =============================================================================

@dataclass(frozen=True)
class FaceDetectionConfig:
    """Face detection configuration."""
    scale_factor: float = 1.3           # Bounding box scale multiplier
    min_face_size: int = 64             # Minimum face size in pixels
    target_size: Tuple[int, int] = (299, 299)  # XceptionNet input size


# =============================================================================
# CLASSIFICATION
# =============================================================================

@dataclass(frozen=True)
class ClassifierConfig:
    """Classifier configuration."""
    model_name: str = 'xception'
    num_classes: int = 2
    dropout: float = 0.5
    input_size: Tuple[int, int] = (299, 299)
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


# =============================================================================
# DECISION AGENT
# =============================================================================

@dataclass(frozen=True)
class DecisionConfig:
    """Decision agent configuration."""
    fake_threshold: float = 0.7        # Score >= this = FAKE
    suspicious_threshold: float = 0.4  # Score >= this (and < fake) = SUSPICIOUS
    min_faces_for_decision: int = 3    # Minimum faces for confident decision
    high_variance_threshold: float = 0.15  # Variance above this = inconsistent


# =============================================================================
# DEFAULTS
# =============================================================================

VIDEO_CONFIG = VideoConfig()
FACE_CONFIG = FaceDetectionConfig()
CLASSIFIER_CONFIG = ClassifierConfig()
DECISION_CONFIG = DecisionConfig()
