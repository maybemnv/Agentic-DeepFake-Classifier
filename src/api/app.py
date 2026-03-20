"""
FastAPI Application
Main application factory with middleware and routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import SlowAPI, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import logging

from .routes import analysis_router, health_router, auth_router
from .security import get_rate_limits_for_tier
from src.core import setup_logging, get_logger, settings

logger = get_logger(__name__)


# Rate limiter
def rate_limit_handler() -> dict:
    """Custom rate limit exceeded handler."""
    return {"error": "Rate limit exceeded", "detail": "Too many requests. Please try again later."}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    setup_logging()
    logger.info("Starting Agentic DeepFake Classifier API", extra={"version": settings.app_version})

    # Initialize rate limiter
    app.state.limiter = SlowAPI(
        key_func=get_remote_address,
        default_limits=[
            f"{settings.rate_limit_per_minute}/minute",
            f"{settings.rate_limit_burst}/burst",
        ],
    )
    app.state.limiter.exceptions = [RateLimitExceeded]
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Load model
    from src.detection import DeepfakeClassifier
    import torch

    use_cuda = settings.use_cuda and torch.cuda.is_available()
    logger.info(f"Loading model with CUDA={use_cuda}")
    app.state.classifier = DeepfakeClassifier(use_cuda=use_cuda)

    logger.info("Model loaded successfully")

    yield

    # Shutdown
    logger.info("Shutting down API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Agentic DeepFake Detector API",
        description="""
        AI-powered deepfake video detection API.

        ## Architecture
        - Model loaded once, cached in memory
        - All requests share the same classifier instance
        - Rate limiting based on user tier

        ## Features
        - Upload videos for deepfake analysis
        - Video quality assessment
        - Comparative analysis (original vs deepfake)
        - Batch processing
        - User authentication with JWT tokens
        - API key management

        ## Authentication
        - Register: `POST /auth/register`
        - Login: `POST /auth/login`
        - API Key: `POST /auth/api-key`

        ## Endpoints
        - `POST /analyze` - Full video analysis with quality check
        - `POST /analyze/quality` - Video quality assessment only
        - `POST /analyze/compare` - Compare two videos
        - `POST /analyze/batch` - Batch process multiple videos
        - `GET /analyze/batch/{job_id}` - Get batch job status
        - `GET /health` - Health check
        """,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=settings.allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(analysis_router)

    return app


app = create_app()
