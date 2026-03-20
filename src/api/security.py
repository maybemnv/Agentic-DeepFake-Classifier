"""
Authentication and Security Module
JWT authentication, API keys, and user management.
"""

from datetime import datetime, timedelta
from typing import Any
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel, Field
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Token Models
# =============================================================================


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Decoded token data."""

    username: str | None = None
    user_id: str | None = None
    tier: str = "free"  # free, premium, enterprise


class APIKey(BaseModel):
    """API key model."""

    key: str
    user_id: str
    tier: str = "free"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    is_active: bool = True


# =============================================================================
# User Models
# =============================================================================


class UserBase(BaseModel):
    """Base user model."""

    username: str
    email: str | None = None
    tier: str = "free"


class UserCreate(UserBase):
    """User creation model."""

    password: str


class User(UserBase):
    """User model."""

    id: str
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


# =============================================================================
# Authentication Functions
# =============================================================================


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        return False


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire, "type": "access"})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm="HS256",
    )

    return encoded_jwt


def create_refresh_token(data: dict[str, Any]) -> str:
    """
    Create a JWT refresh token.

    Args:
        data: Data to encode in the token

    Returns:
        Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm="HS256",
    )

    return encoded_jwt


def decode_token(token: str) -> TokenData | None:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        tier: str = payload.get("tier", "free")

        if username is None:
            return None

        return TokenData(username=username, user_id=user_id, tier=tier)

    except JWTError as e:
        logger.warning(f"Token decode error: {e}")
        return None


def generate_api_key() -> str:
    """
    Generate a secure API key.

    Returns:
        Random API key string
    """
    import secrets

    return f"dfk_{secrets.token_urlsafe(32)}"


def create_api_key_for_user(user_id: str, tier: str = "free") -> APIKey:
    """
    Create an API key for a user.

    Args:
        user_id: User ID
        tier: User tier (affects rate limits)

    Returns:
        APIKey object
    """
    return APIKey(
        key=generate_api_key(),
        user_id=user_id,
        tier=tier,
    )


# =============================================================================
# Rate Limit Tiers
# =============================================================================

RATE_LIMIT_TIERS: dict[str, dict[str, int]] = {
    "free": {
        "per_minute": 5,
        "per_hour": 50,
        "per_day": 200,
        "max_upload_mb": 100,
        "max_frames": 100,
    },
    "premium": {
        "per_minute": 20,
        "per_hour": 200,
        "per_day": 1000,
        "max_upload_mb": 500,
        "max_frames": 500,
    },
    "enterprise": {
        "per_minute": 100,
        "per_hour": 1000,
        "per_day": 10000,
        "max_upload_mb": 2000,
        "max_frames": 0,  # Unlimited
    },
}


def get_rate_limits_for_tier(tier: str) -> dict[str, int]:
    """Get rate limits for a user tier."""
    return RATE_LIMIT_TIERS.get(tier, RATE_LIMIT_TIERS["free"])
