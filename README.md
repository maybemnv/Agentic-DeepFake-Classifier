# Agentic Deepfake Classifier

An autonomous deepfake detection system utilizing XceptionNet and agentic reasoning to analyze video authenticity.

[![CI/CD Pipeline](https://github.com/yourusername/Agentic-DeepFake-Classifier/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/yourusername/Agentic-DeepFake-Classifier/actions/workflows/ci-cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## 📋 Table of Contents

- [System Overview](#-system-overview)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Architecture](#-architecture)
- [Development](#-development)
- [Deployment](#-deployment)
- [Benchmarks](#-benchmarks)
- [License](#-license)

## System Overview

This system implements a modular pipeline for deepfake detection, focusing on explainability and temporal consistency:

1. **Video Processing**: Efficient frame extraction at configurable sample rates using OpenCV.
2. **Face Detection**: Face identification and cropping using dlib.
3. **Quality Assessment**: Pre-analysis video quality evaluation.
4. **Inference Engine**: PyTorch implementation of XceptionNet trained on FaceForensics++ (c23 compression).
5. **Agentic Analysis**:
   - **Decision Agent**: Aggregates per-frame probabilities and applies temporal logic to determine a final verdict.
   - **Cognitive Agent**: Synthesizes technical metrics into human-readable explanations.

### Technology Stack

- **Runtime**: Python 3.10+
- **Inference**: PyTorch (CUDA supported) / ONNX Runtime
- **API**: FastAPI with rate limiting
- **Interface**: Streamlit
- **Vision**: OpenCV, dlib
- **Database**: PostgreSQL (optional)
- **Cache**: Redis (optional)
- **Container**: Docker, Docker Compose

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/Agentic-DeepFake-Classifier.git
cd Agentic-DeepFake-Classifier

# Copy environment file
cp .env.example .env

# Start with Docker Compose
docker-compose up -d

# Access the UI
open http://localhost:8501

# Access the API docs
open http://localhost:8000/docs
```

## Installation

### Using uv (Recommended)

This project utilizes `uv` for fast dependency management.

```bash
# Clone repository
git clone https://github.com/yourusername/Agentic-DeepFake-Classifier.git
cd Agentic-DeepFake-Classifier

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### With Docker

```bash
# Build and run
docker-compose up -d

# Or build manually
docker build -t deepfake-classifier .
docker run -p 8000:8000 deepfake-classifier api
```

## Usage

### Web Interface

Launch the interactive dashboard for video analysis.

```bash
streamlit run frontend/app.py
```

Access the UI at `http://localhost:8501`.

### REST API

Start the backend server for programmatic integration.

```bash
uvicorn src.api.app:app --reload
```

API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

#### Example API Usage

```python
import requests

# Register a user
response = requests.post("http://localhost:8000/auth/register", data={
    "username": "demo",
    "password": "demo123"
})
token = response.json()["access_token"]

# Analyze a video
with open("video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        headers={"Authorization": f"Bearer {token}"}
    )
    result = response.json()
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.1%}")
```

### Command Line Interface

Analyze videos via the terminal.

```bash
# Output summary to console
python main.py --video data/sample.mp4

# Save detailed JSON report
python main.py --video data/sample.mp4 --output report.json

# Quick analysis (first 5 frames)
python main.py --video data/sample.mp4 --quick

# Use GPU
python main.py --video data/sample.mp4 --cuda
```

### Batch Processing

```python
import requests

# Upload multiple videos
files = [
    ("files", ("video1.mp4", open("video1.mp4", "rb"))),
    ("files", ("video2.mp4", open("video2.mp4", "rb"))),
    ("files", ("video3.mp4", open("video3.mp4", "rb"))),
]

response = requests.post("http://localhost:8000/analyze/batch", files=files)
job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/analyze/batch/{job_id}")
print(status.json())
```

### Comparative Analysis

```python
import requests

# Compare two videos
with open("original.mp4", "rb") as f1, open("suspected.mp4", "rb") as f2:
    response = requests.post(
        "http://localhost:8000/analyze/compare",
        files={
            "video1": f1,
            "video2": f2
        },
        data={
            "video1_description": "Original video",
            "video2_description": "Suspected deepfake"
        }
    )
    result = response.json()
    print(f"Similarity: {result['similarity_score']:.1%}")
    print(f"Conclusion: {result['conclusion']}")
```

## API Reference

### Authentication Endpoints

| Method | Endpoint            | Description            |
| ------ | ------------------- | ---------------------- |
| POST   | `/auth/register`    | Register new user      |
| POST   | `/auth/login`       | Login with credentials |
| POST   | `/auth/refresh`     | Refresh access token   |
| POST   | `/auth/api-key`     | Create API key         |
| GET    | `/auth/me`          | Get current user       |
| GET    | `/auth/rate-limits` | Get rate limits        |

### Analysis Endpoints

| Method | Endpoint                  | Description          |
| ------ | ------------------------- | -------------------- |
| POST   | `/analyze`                | Analyze single video |
| POST   | `/analyze/quality`        | Check video quality  |
| POST   | `/analyze/compare`        | Compare two videos   |
| POST   | `/analyze/batch`          | Batch process videos |
| GET    | `/analyze/batch/{job_id}` | Get batch job status |
| GET    | `/health`                 | Health check         |

### User Tiers & Rate Limits

| Tier       | Requests/min | Max Upload | Max Frames | Batch Limit |
| ---------- | ------------ | ---------- | ---------- | ----------- |
| Free       | 5            | 100 MB     | 100        | 10 videos   |
| Premium    | 20           | 500 MB     | 500        | 50 videos   |
| Enterprise | 100          | 2000 MB    | Unlimited  | Unlimited   |

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Application
APP_NAME=Agentic DeepFake Classifier
LOG_LEVEL=INFO
LOG_FORMAT=json

# API
API_HOST=0.0.0.0
API_PORT=8000

# Authentication
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate Limiting
RATE_LIMIT_PER_MINUTE=10

# Model
MODEL_PATH=model/ffpp_c23.pth
USE_CUDA=true

# Detection Thresholds
FAKE_THRESHOLD=0.7
SUSPICIOUS_THRESHOLD=0.4
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Agentic DeepFake Classifier            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Frontend   │  │     API     │  │      Workers        │  │
│  │  Streamlit  │◄─┤   FastAPI   │◄─┤  Background Jobs    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                │                    │              │
│         │                ▼                    │              │
│         │      ┌──────────────────┐          │              │
│         │      │   Authentication │          │              │
│         │      │   Rate Limiting  │          │              │
│         │      └──────────────────┘          │              │
│         │                │                    │              │
│         ▼                ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Detection Pipeline                      │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐  │    │
│  │  │ Quality │→ │  Face   │→ │  Xception│→ │Decision│  │    │
│  │  │  Check  │  │ Detect  │  │  Net    │  │ Agent  │  │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Redis     │  │  PostgreSQL │  │   Model Cache       │  │
│  │   (Cache)   │  │   (Data)    │  │   (Weights)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
Agentic-DeepFake-Classifier/
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analysis.py      # Analysis endpoints
│   │   │   ├── auth.py          # Authentication endpoints
│   │   │   └── health.py        # Health check
│   │   ├── app.py               # FastAPI application
│   │   ├── schemas.py           # Pydantic models
│   │   ├── security.py          # Auth & rate limiting
│   │   └── dependencies.py      # DI dependencies
│   ├── core/
│   │   ├── models.py            # Core Pydantic models
│   │   ├── config.py            # Settings management
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── logging_config.py    # Structured logging
│   ├── detection/
│   │   ├── classifier.py        # PyTorch classifier
│   │   ├── onnx_classifier.py   # ONNX optimized classifier
│   │   ├── face.py              # Face detection
│   │   ├── video.py             # Video processing
│   │   └── quality.py           # Quality assessment
│   ├── agents/
│   │   ├── decision.py          # Decision agent
│   │   └── cognitive.py         # Explanation agent
│   ├── pipeline/
│   │   ├── analysis.py          # Analysis pipeline
│   │   └── detection.py         # Detection pipeline
│   └── workers/
│       └── batch_worker.py      # Background worker
├── frontend/
│   └── app.py                   # Streamlit UI
├── benchmarks/
│   └── performance.py           # Performance benchmarks
├── tests/
│   ├── test_core.py             # Core model tests
│   ├── test_api.py              # API endpoint tests
│   └── test_quality.py          # Quality assessment tests
├── model/
│   └── ffpp_c23.pth             # Pre-trained weights
├── .env.example                 # Environment template
├── docker-compose.yml           # Multi-service setup
└── pyproject.toml               # Project dependencies
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Linting
ruff check src/ frontend/

# Formatting
ruff format src/ frontend/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

### Running Benchmarks

```python
from src.detection import DeepfakeClassifier, ONNXClassifier
from src.pipeline import DeepfakeAnalyzer
from benchmarks.performance import run_all_benchmarks

# Initialize
classifier = DeepfakeClassifier(use_cuda=True)
analyzer = DeepfakeAnalyzer(classifier)

# Run benchmarks
results = run_all_benchmarks(
    classifier,
    analyzer,
    test_video="data/sample.mp4",
    output_dir="benchmark_results"
)
```

## Deployment

### Docker Compose (Production)

```bash
# Set secret key
export SECRET_KEY=$(openssl rand -hex 32)

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Scale workers
docker-compose up -d --scale worker=3
```

### Kubernetes (Coming Soon)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepfake-api
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: api
          image: ghcr.io/yourusername/agentic-deepfake-classifier:latest
          resources:
            limits:
              nvidia.com/gpu: 1
```

### Cloud Deployment

#### AWS

```bash
# Using ECS with GPU support
aws ecs create-cluster --cluster-name deepfake-cluster

# Deploy with CloudFormation
aws cloudformation deploy \
  --template-file aws/cloudformation.yml \
  --stack-name deepfake-classifier
```

#### GCP

```bash
# Using Cloud Run with GPU
gcloud run deploy deepfake-classifier \
  --image gcr.io/your-project/deepfake-classifier \
  --gpu-type nvidia-tesla-t4 \
  --gpu-count 1
```

## Benchmarks

### Performance Comparison

| Model            | Device | Avg Inference | Throughput |
| ---------------- | ------ | ------------- | ---------- |
| PyTorch Xception | CPU    | 45ms          | 22 fps     |
| ONNX Xception    | CPU    | 18ms          | 55 fps     |
| PyTorch Xception | GPU    | 8ms           | 125 fps    |
| ONNX Xception    | GPU    | 5ms           | 200 fps    |

### Video Analysis Performance

| Video Length | Processing Time | Speed Factor |
| ------------ | --------------- | ------------ |
| 10 seconds   | 3 seconds       | 3.3x faster  |
| 30 seconds   | 8 seconds       | 3.75x faster |
| 60 seconds   | 15 seconds      | 4x faster    |

## License

The source code is released under the MIT License.

This work builds upon the **FaceForensics++** dataset and benchmark. Usage of the pre-trained weights is subject to the FaceForensics++ license terms (non-commercial research use).

- **FaceForensics++**: Rössler et al. (ICCV 2019)
- **Xception**: Cholull (CVPR 2017)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
