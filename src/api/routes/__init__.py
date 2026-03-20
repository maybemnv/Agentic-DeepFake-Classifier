"""
Routes Module
"""

from .analysis import router as analysis_router
from .health import router as health_router
from .auth import router as auth_router

__all__ = ["analysis_router", "health_router", "auth_router"]
