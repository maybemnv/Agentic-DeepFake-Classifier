# Makefile for Agentic DeepFake Classifier
# Common development and deployment commands

.PHONY: help install dev test lint format typecheck security bench docker docker-up docker-down clean

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Agentic DeepFake Classifier - Makefile Commands"
	@echo ""
	@echo "Installation:"
	@echo "  install        - Install production dependencies"
	@echo "  dev            - Install with development dependencies"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test           - Run all tests"
	@echo "  test-cov       - Run tests with coverage report"
	@echo "  lint           - Run linter (ruff)"
	@echo "  format         - Format code (ruff)"
	@echo "  typecheck      - Run type checker (mypy)"
	@echo "  security       - Run security scans"
	@echo "  check          - Run all quality checks"
	@echo ""
	@echo "Development:"
	@echo "  server         - Start API server (development)"
	@echo "  ui             - Start Streamlit UI"
	@echo "  worker         - Start background worker"
	@echo ""
	@echo "Docker:"
	@echo "  docker         - Build Docker image"
	@echo "  docker-up      - Start all services (docker-compose)"
	@echo "  docker-down    - Stop all services"
	@echo "  docker-logs    - View logs"
	@echo "  docker-clean   - Remove containers and volumes"
	@echo ""
	@echo "Benchmarks:"
	@echo "  bench          - Run performance benchmarks"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          - Remove build artifacts"
	@echo "  clean-all      - Remove all generated files"

# =============================================================================
# Installation
# =============================================================================

install:
	@echo "Installing production dependencies..."
	uv sync --frozen

dev:
	@echo "Installing development dependencies..."
	uv sync --all-extras

# =============================================================================
# Testing & Quality
# =============================================================================

test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

lint:
	@echo "Running linter..."
	ruff check src/ frontend/ benchmarks/

format:
	@echo "Formatting code..."
	ruff format src/ frontend/ benchmarks/

typecheck:
	@echo "Running type checker..."
	mypy src/ --ignore-missing-imports

security:
	@echo "Running security scans..."
	bandit -r src/ -f json -o bandit-report.json || true
	safety check --json > safety-report.json || true
	@echo "Reports: bandit-report.json, safety-report.json"

check: lint format typecheck security
	@echo "All quality checks completed"

# =============================================================================
# Development Servers
# =============================================================================

server:
	@echo "Starting API server..."
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

ui:
	@echo "Starting Streamlit UI..."
	streamlit run frontend/app.py --server.port 8501

worker:
	@echo "Starting background worker..."
	python -m src.workers.batch_worker

# =============================================================================
# Docker
# =============================================================================

docker:
	@echo "Building Docker image..."
	docker build -t deepfake-classifier:latest .

docker-up:
	@echo "Starting all services..."
	docker-compose up -d

docker-down:
	@echo "Stopping all services..."
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	@echo "Removing containers and volumes..."
	docker-compose down -v --remove-orphans

# =============================================================================
# Benchmarks
# =============================================================================

bench:
	@echo "Running performance benchmarks..."
	python -m benchmarks.run

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ wheels/
	rm -rf benchmark_results/
	rm -rf uploads/
	rm -rf .cache/
	@echo "Cleanup completed"

clean-all: clean
	@echo "Removing all generated files..."
	rm -rf .venv/
	rm -rf node_modules/
	rm -f bandit-report.json safety-report.json
	@echo "Full cleanup completed"

# =============================================================================
# Database (PostgreSQL)
# =============================================================================

db-up:
	@echo "Starting database..."
	docker-compose up -d postgres

db-down:
	@echo "Stopping database..."
	docker-compose down postgres

db-shell:
	@echo "Connecting to database..."
	docker-compose exec postgres psql -U deepfake -d deepfake_db

# =============================================================================
# Redis
# =============================================================================

redis-up:
	@echo "Starting Redis..."
	docker-compose up -d redis

redis-down:
	@echo "Stopping Redis..."
	docker-compose down redis

redis-cli:
	@echo "Connecting to Redis..."
	docker-compose exec redis redis-cli

# =============================================================================
# Model Management
# =============================================================================

model-info:
	@echo "Model Information:"
	@ls -lh model/

model-download:
	@echo "Downloading model weights..."
	@echo "Please download ffpp_c23.pth from the official source"
	@echo "Place it in the model/ directory"
