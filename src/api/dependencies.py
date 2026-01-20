"""
API Dependencies
Shared dependencies for API routes.
"""

from functools import lru_cache
from typing import Optional
import logging

from ..detection import DeepfakeClassifier

logger = logging.getLogger(__name__)


class ClassifierService:
    """
    Singleton service for the DeepfakeClassifier.
    Prevents re-loading the model on every request.
    """

    _instance: Optional[DeepfakeClassifier] = None

    @classmethod
    def get_classifier(
        cls, weights_path: Optional[str] = None, use_cuda: bool = False
    ) -> DeepfakeClassifier:
        """Get or create the classifier instance."""
        if cls._instance is None:
            logger.info("Loading DeepfakeClassifier (first request)...")
            cls._instance = DeepfakeClassifier(
                use_cuda=use_cuda
            )
            logger.info("DeepfakeClassifier loaded and cached!")
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the classifier (for testing)."""
        cls._instance = None


def get_classifier() -> DeepfakeClassifier:
    """Dependency to get the shared classifier instance."""
    return ClassifierService.get_classifier()
