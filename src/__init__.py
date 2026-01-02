"""
Source Package
Agentic Deepfake Classifier
"""

# Main public API
from .pipeline import DeepfakeAnalyzer, analyze_video

# Core exports
from .core import (
    Verdict,
    AnalysisResult,
    VideoAnalysis,
    DecisionResult
)

__version__ = "1.0.0"

__all__ = [
    'DeepfakeAnalyzer',
    'analyze_video',
    'Verdict',
    'AnalysisResult',
    'VideoAnalysis',
    'DecisionResult',
]
