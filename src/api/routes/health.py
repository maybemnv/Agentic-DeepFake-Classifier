"""
Health Routes
System health and status endpoints.
"""

from fastapi import APIRouter
from ..schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is running."
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Root endpoint",
    include_in_schema=False
)
async def root():
    """Root endpoint redirects to health."""
    return HealthResponse(status="healthy", version="1.0.0")
