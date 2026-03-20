"""
Test API Endpoints
Tests for FastAPI routes.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestAnalyzeEndpoint:
    """Test video analysis endpoint."""

    def test_analyze_missing_file(self, client):
        response = client.post("/analyze")
        assert response.status_code == 422  # Validation error

    def test_analyze_invalid_type(self, client, tmp_path):
        # Create a fake text file
        fake_file = tmp_path / "fake.txt"
        fake_file.write_text("not a video")

        with open(fake_file, "rb") as f:
            response = client.post(
                "/analyze",
                files={"file": ("fake.txt", f, "text/plain")},
            )
        assert response.status_code == 400


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_register_user(self, client):
        response = client.post(
            "/auth/register",
            json={
                "username": "testuser",
                "password": "testpass123",
                "email": "test@example.com",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_login_user(self, client):
        # First register
        client.post(
            "/auth/register",
            json={
                "username": "loginuser",
                "password": "testpass123",
            },
        )

        # Then login
        response = client.post(
            "/auth/login",
            data={
                "username": "loginuser",
                "password": "testpass123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    def test_login_wrong_password(self, client):
        # First register
        client.post(
            "/auth/register",
            json={
                "username": "wrongpassuser",
                "password": "testpass123",
            },
        )

        # Then login with wrong password
        response = client.post(
            "/auth/login",
            data={
                "username": "wrongpassuser",
                "password": "wrongpassword",
            },
        )
        assert response.status_code == 401

    def test_duplicate_username(self, client):
        # Register first user
        response1 = client.post(
            "/auth/register",
            json={
                "username": "duplicateuser",
                "password": "testpass123",
            },
        )
        assert response1.status_code == 200

        # Try to register same username
        response2 = client.post(
            "/auth/register",
            json={
                "username": "duplicateuser",
                "password": "testpass456",
            },
        )
        assert response2.status_code == 400


class TestRateLimits:
    """Test rate limit endpoints."""

    def test_get_rate_limits(self, client):
        # Register and login
        reg_response = client.post(
            "/auth/register",
            json={
                "username": "ratelimituser",
                "password": "testpass123",
            },
        )
        token = reg_response.json()["access_token"]

        # Get rate limits
        response = client.get(
            "/auth/rate-limits",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "tier" in data
        assert "per_minute" in data


class TestAPIKey:
    """Test API key functionality."""

    def test_create_api_key(self, client):
        # Register and login
        reg_response = client.post(
            "/auth/register",
            json={
                "username": "apiuser",
                "password": "testpass123",
            },
        )
        token = reg_response.json()["access_token"]

        # Create API key
        response = client.post(
            "/auth/api-key",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "key" in data
        assert data["key"].startswith("dfk_")

    def test_get_current_user(self, client):
        # Register and login
        reg_response = client.post(
            "/auth/register",
            json={
                "username": "meuser",
                "password": "testpass123",
                "email": "me@example.com",
            },
        )
        token = reg_response.json()["access_token"]

        # Get current user
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "meuser"
