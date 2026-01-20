"""
Core Module
Contains models, configuration, and exceptions.
"""

from .models import (
    Verdict,
    VideoMetadata,
    FaceResult,
    ClassificationResult,
    FrameAnalysis,
    VideoAnalysis,
    DecisionResult,
    CognitiveResponse,
    AnalysisResult
)

from .config import (
    PROJECT_ROOT,
    MODEL_DIR,
    VideoConfig,
    FaceDetectionConfig,
    ClassifierConfig,
    DecisionConfig,
    VIDEO_CONFIG,
    FACE_CONFIG,
    CLASSIFIER_CONFIG,
    DECISION_CONFIG
)

from .exceptions import (
    DeepfakeDetectorError,
    VideoError,
    VideoNotFoundError,
    VideoFormatError,
    VideoCorruptedError,
    FaceDetectionError,
    NoFacesDetectedError,
    ClassifierError,
    ModelNotFoundError,
    ModelLoadError
)

__all__ = [
    # Models
    'Verdict',
    'VideoMetadata',
    'FaceResult',
    'ClassificationResult',
    'FrameAnalysis',
    'VideoAnalysis',
    'DecisionResult',
    'CognitiveResponse',
    'AnalysisResult',
    # Config
    'PROJECT_ROOT',
    'MODEL_DIR',
    'VideoConfig',
    'FaceDetectionConfig',
    'ClassifierConfig',
    'DecisionConfig',
    'VIDEO_CONFIG',
    'FACE_CONFIG',
    'CLASSIFIER_CONFIG',
    'DECISION_CONFIG',
    # Exceptions
    'DeepfakeDetectorError',
    'VideoError',
    'VideoNotFoundError',
    'VideoFormatError',
    'VideoCorruptedError',
    'FaceDetectionError',
    'NoFacesDetectedError',
    'ClassifierError',
    'ModelNotFoundError',
    'ModelLoadError',
]
