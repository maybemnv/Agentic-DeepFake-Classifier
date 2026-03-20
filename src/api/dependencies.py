"""
API Dependencies
Shared dependencies for API routes.
"""

from typing import Any
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..detection import DeepfakeClassifier
from .security import decode_token, users_db, User

security = HTTPBearer(auto_error=False)


def get_classifier(request: Request) -> DeepfakeClassifier:
    """Dependency to get the shared classifier instance from app state."""
    return request.app.state.classifier


async def get_current_user_from_auth(
    credentials: HTTPAuthorizationCredentials | None = None,
) -> User | None:
    """
    Get current user from authentication credentials.

    Returns None if not authenticated (for optional auth endpoints).
    Raises HTTPException if invalid credentials provided.
    """
    if credentials is None:
        return None

    try:
        token_data = decode_token(credentials.credentials)
        if token_data is None or token_data.user_id is None:
            return None

        user = users_db.get(token_data.user_id)
        if user is None or not user.is_active:
            return None

        return user

    except Exception:
        return None


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User:
    """
    Require authentication for an endpoint.

    Raises HTTPException if not authenticated.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = decode_token(credentials.credentials)
    if token_data is None or token_data.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = users_db.get(token_data.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return user


async def get_api_key_user(
    api_key: str,
) -> User | None:
    """
    Get user by API key.

    Returns None if API key is invalid.
    """
    from .security import api_keys_db

    api_key_obj = api_keys_db.get(api_key)
    if api_key_obj is None or not api_key_obj.is_active:
        return None

    # Check expiration
    from datetime import datetime

    if api_key_obj.expires_at and datetime.utcnow() > api_key_obj.expires_at:
        return None

    return users_db.get(api_key_obj.user_id)
