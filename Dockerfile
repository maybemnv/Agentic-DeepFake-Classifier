# Multi-stage Dockerfile for Agentic DeepFake Classifier

# =============================================================================
# Base Image
# =============================================================================
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# =============================================================================
# Dependencies Stage
# =============================================================================
FROM base as dependencies

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install dependencies
RUN uv sync --frozen 2>/dev/null || uv pip install -r requirements.txt

# =============================================================================
# Production Stage
# =============================================================================
FROM base as production

# Install dependencies from cache
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/.cache /app/logs

# Set permissions
RUN chmod +x /app/entrypoint.sh 2>/dev/null || true

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
if [ "$1" = "api" ]; then\n\
    echo "Starting API server..."\n\
    exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000\n\
elif [ "$1" = "ui" ]; then\n\
    echo "Starting Streamlit UI..."\n\
    exec streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0\n\
elif [ "$1" = "worker" ]; then\n\
    echo "Starting background worker..."\n\
    exec python -m src.workers.batch_worker\n\
elif [ "$1" = "dev" ]; then\n\
    echo "Starting development server..."\n\
    exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload\n\
else\n\
    echo "Usage: entrypoint.sh [api|ui|worker|dev]"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]

# =============================================================================
# Development Stage
# =============================================================================
FROM production as development

# Install development dependencies
RUN uv pip install pytest pytest-asyncio pytest-cov httpx ruff mypy bandit

# Set development environment
ENV APP_ENV=development \
    LOG_LEVEL=DEBUG \
    LOG_FORMAT=text \
    API_RELOAD=true

CMD ["dev"]
