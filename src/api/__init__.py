"""
API Module
FastAPI endpoints for deepfake detection.
"""

from .app import app, create_app
from .schemas import (
    AnalysisResponse,
    QuickCheckResponse,
    HealthResponse,
    ErrorResponse
)

__all__ = [
    'app',
    'create_app',
    'AnalysisResponse',
    'QuickCheckResponse',
    'HealthResponse',
    'ErrorResponse'
]
