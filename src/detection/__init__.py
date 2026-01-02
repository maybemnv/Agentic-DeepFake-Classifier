"""
Detection Module
Video processing, face detection, and deepfake classification.
"""

from .video import VideoProcessor
from .face import FaceDetector
from .classifier import DeepfakeClassifier

__all__ = ['VideoProcessor', 'FaceDetector', 'DeepfakeClassifier']
