"""
Core Module
Contains models, configuration, exceptions, and logging.
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
    AnalysisResult,
    VideoQualityMetrics,
    BatchJobStatus,
    BatchJobInfo,
    ComparativeAnalysisResult,
)

from .config import (
    Settings,
    settings,
    get_settings,
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
    ModelLoadError,
)

from .logging_config import setup_logging, get_logger

__all__ = [
    # Models
    "Verdict",
    "VideoMetadata",
    "FaceResult",
    "ClassificationResult",
    "FrameAnalysis",
    "VideoAnalysis",
    "DecisionResult",
    "CognitiveResponse",
    "AnalysisResult",
    "VideoQualityMetrics",
    "BatchJobStatus",
    "BatchJobInfo",
    "ComparativeAnalysisResult",
    # Config
    "Settings",
    "settings",
    "get_settings",
    # Exceptions
    "DeepfakeDetectorError",
    "VideoError",
    "VideoNotFoundError",
    "VideoFormatError",
    "VideoCorruptedError",
    "FaceDetectionError",
    "NoFacesDetectedError",
    "ClassifierError",
    "ModelNotFoundError",
    "ModelLoadError",
    # Logging
    "setup_logging",
    "get_logger",
]
