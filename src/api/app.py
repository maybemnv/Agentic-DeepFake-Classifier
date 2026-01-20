"""
FastAPI Application
Main API application setup.

Model loaded once on first request, shared by all subsequent requests.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import analysis_router, health_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Agentic Deepfake Detector API",
        description="""
        AI-powered deepfake video detection API.

        ## Architecture
        - Model loaded once, cached in memory
        - All requests share the same classifier instance
        - No per-request model loading overhead

        ## Features
        - Upload videos for deepfake analysis
        - Get confidence scores and explanations
        - Quick check mode for fast results

        ## Endpoints
        - `POST /analyze` - Full video analysis (recommended)
        - `POST /analyze/video` - Full video analysis (legacy)
        - `GET /health` - Health check
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.on_event("startup")
    async def startup_event():
        from ..detection import DeepfakeClassifier
        import torch
        
        # Load model once
        # Check for CUDA availability for auto-selection, or force if env var set?
        # User didn't specify, so let's default to safe auto-detect.
        use_cuda = torch.cuda.is_available()
        app.state.classifier = DeepfakeClassifier(use_cuda=use_cuda)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(analysis_router)

    return app


# Create app instance
app = create_app()
