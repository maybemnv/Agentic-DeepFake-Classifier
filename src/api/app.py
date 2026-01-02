"""
FastAPI Application
Main API application setup.
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
        
        ## Features
        - Upload videos for deepfake analysis
        - Get confidence scores and explanations
        - Quick check mode for fast results
        
        ## Endpoints
        - `POST /analyze/video` - Full video analysis
        - `POST /analyze/quick` - Quick 5-frame check
        - `GET /health` - Health check
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(analysis_router)
    
    return app


# Create app instance
app = create_app()
