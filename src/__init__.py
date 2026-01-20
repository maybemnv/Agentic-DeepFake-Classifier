"""
Source Package
Agentic Deepfake Classifier
"""

# Main public API
from .pipeline import DeepfakeAnalyzer, analyze_video

# Detection exports
from .detection import DeepfakeClassifier

# Core exports
from .core import Verdict, AnalysisResult, VideoAnalysis, DecisionResult

__version__ = "1.0.0"

__all__ = [
    "DeepfakeAnalyzer",
    "analyze_video",
    "DeepfakeClassifier",
    "Verdict",
    "AnalysisResult",
    "VideoAnalysis",
    "DecisionResult",
]
