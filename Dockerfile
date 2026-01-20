# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install dependencies
RUN uv sync --frozen || uv pip install -r requirements.txt

# Copy project files
COPY . .

# Expose ports for API and Streamlit
EXPOSE 8000
EXPOSE 8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000\n\
elif [ "$1" = "ui" ]; then\n\
    exec streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0\n\
else\n\
    echo "Usage: docker run <image> [api|ui]"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
