"""
API Dependencies
Shared dependencies for API routes.
"""

from functools import lru_cache
from typing import Optional
import logging

from ..pipeline import DeepfakeAnalyzer
from ..core import DecisionConfig

logger = logging.getLogger(__name__)


class AnalyzerService:
    """
    Singleton service for the analyzer.
    Prevents re-loading the model on every request.
    """
    _instance: Optional[DeepfakeAnalyzer] = None
    
    @classmethod
    def get_analyzer(
        cls,
        sample_rate: float = 1.0,
        max_frames: Optional[int] = None,
        fake_threshold: float = 0.7,
        suspicious_threshold: float = 0.4
    ) -> DeepfakeAnalyzer:
        """Get or create the analyzer instance."""
        if cls._instance is None:
            logger.info("Creating new DeepfakeAnalyzer instance")
            cls._instance = DeepfakeAnalyzer(
                sample_rate=sample_rate,
                max_frames=max_frames,
                fake_threshold=fake_threshold,
                suspicious_threshold=suspicious_threshold
            )
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the analyzer (for testing)."""
        cls._instance = None


def get_analyzer() -> DeepfakeAnalyzer:
    """Dependency to get the analyzer."""
    return AnalyzerService.get_analyzer()
