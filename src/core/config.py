"""
Configuration Module
Central place for all configuration values using Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Literal
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =============================================================================
    # Application Settings
    # =============================================================================
    app_name: str = Field(
        default="Agentic DeepFake Classifier", description="Application name"
    )
    app_version: str = Field(default="1.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="Log format"
    )

    # =============================================================================
    # API Settings
    # =============================================================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_reload: bool = Field(default=False, description="Auto-reload for development")

    # =============================================================================
    # Authentication
    # =============================================================================
    secret_key: str = Field(
        default="change-me-in-production", description="Secret key for JWT tokens"
    )
    access_token_expire_minutes: int = Field(
        default=30, ge=1, description="Access token expiration in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7, ge=1, description="Refresh token expiration in days"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    # =============================================================================
    # Rate Limiting
    # =============================================================================
    rate_limit_per_minute: int = Field(
        default=10, ge=1, description="Rate limit per minute"
    )
    rate_limit_burst: int = Field(default=20, ge=1, description="Rate limit burst")

    # =============================================================================
    # Model Settings
    # =============================================================================
    model_path: str = Field(
        default="model/ffpp_c23.pth", description="Path to model weights"
    )
    use_cuda: bool = Field(default=True, description="Use CUDA for inference")
    batch_size: int = Field(default=32, ge=1, description="Batch size for inference")
    input_size: int = Field(default=299, ge=32, description="Model input size")

    # =============================================================================
    # Detection Thresholds
    # =============================================================================
    fake_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Threshold for FAKE verdict"
    )
    suspicious_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Threshold for SUSPICIOUS verdict"
    )
    min_faces_for_decision: int = Field(
        default=3, ge=1, description="Minimum faces for confident decision"
    )

    # =============================================================================
    # Video Processing
    # =============================================================================
    sample_rate: float = Field(
        default=1.0, ge=0.1, description="Frame sampling rate (fps)"
    )
    max_frames: int = Field(default=0, ge=0, description="Max frames (0 = unlimited)")
    video_cache_dir: str = Field(
        default=".cache/videos", description="Video cache directory"
    )

    # =============================================================================
    # Redis
    # =============================================================================
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    redis_db: int = Field(default=0, ge=0, le=15, description="Redis database")
    redis_password: str | None = Field(default=None, description="Redis password")

    # =============================================================================
    # Database
    # =============================================================================
    database_url: str = Field(
        default="sqlite+aiosqlite:///./deepfake_analyzer.db", description="Database URL"
    )

    # =============================================================================
    # Storage
    # =============================================================================
    upload_dir: str = Field(default="./uploads", description="Upload directory")
    max_upload_size_mb: int = Field(
        default=500, ge=1, description="Max upload size in MB"
    )

    # =============================================================================
    # CORS
    # =============================================================================
    allowed_origins: str = Field(
        default="http://localhost:8501,http://localhost:3000",
        description="Comma-separated allowed origins",
    )
    allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS"
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate that secret key is changed in production."""
        if (
            v == "change-me-in-production"
            and os.getenv("APP_ENV", "development") == "production"
        ):
            raise ValueError(
                "SECRET_KEY must be changed in production. Generate one with: openssl rand -hex 32"
            )
        return v

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse allowed origins as a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance (for dependency injection)."""
    return settings
