"""
Authentication Routes
Endpoints for user registration, login, and API key management.
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status, Security, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from ..security import (
    Token,
    UserCreate,
    User,
    APIKey,
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    create_api_key_for_user,
    get_rate_limits_for_tier,
)
from ..schemas import APIKeyResponse, RateLimitInfo, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)

# In-memory user store (replace with database in production)
users_db: dict[str, User] = {}
api_keys_db: dict[str, APIKey] = {}


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> User:
    """Get current authenticated user from JWT token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
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


async def get_user_by_api_key(api_key: str) -> User | None:
    """Get user by API key."""
    api_key_obj = api_keys_db.get(api_key)
    if api_key_obj is None or not api_key_obj.is_active:
        return None

    # Check expiration
    if api_key_obj.expires_at and datetime.utcnow() > api_key_obj.expires_at:
        return None

    return users_db.get(api_key_obj.user_id)


@router.post("/register", response_model=Token, summary="Register a new user")
async def register(user_data: UserCreate):
    """
    Register a new user account.

    - **username**: Unique username
    - **email**: Optional email address
    - **password**: Password (will be hashed)
    """
    # Check if user exists
    for user in users_db.values():
        if user.username == user_data.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken",
            )

    # Create user
    user_id = str(uuid.uuid4())
    user = User(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        tier=user_data.tier,
        hashed_password=get_password_hash(user_data.password),
    )

    users_db[user_id] = user
    logger.info(f"User registered: {user.username}")

    # Generate tokens
    token_data = {"sub": user.username, "user_id": user_id, "tier": user.tier}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/login", response_model=Token, summary="Login user")
async def login(username: str = Form(...), password: str = Form(...)):
    """
    Login with username and password.

    Returns access and refresh tokens.
    """
    # Find user
    user = None
    for u in users_db.values():
        if u.username == username:
            user = u
            break

    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    logger.info(f"User logged in: {user.username}")

    # Generate tokens
    token_data = {"sub": user.username, "user_id": user.id, "tier": user.tier}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=Token, summary="Refresh access token")
async def refresh_token(refresh_token: str = Form(...)):
    """Refresh access token using refresh token."""
    token_data = decode_token(refresh_token)

    if token_data is None or token_data.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user = users_db.get(token_data.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Generate new tokens
    new_token_data = {"sub": user.username, "user_id": user.id, "tier": user.tier}
    new_access_token = create_access_token(new_token_data)
    new_refresh_token = create_refresh_token(new_token_data)

    return Token(access_token=new_access_token, refresh_token=new_refresh_token)


@router.post("/api-key", response_model=APIKeyResponse, summary="Create API key")
async def create_api_key(current_user: User = Depends(get_current_user)):
    """
    Create a new API key for the authenticated user.

    API keys are useful for server-to-server authentication.
    """
    api_key = create_api_key_for_user(current_user.id, current_user.tier)
    api_keys_db[api_key.key] = api_key

    logger.info(f"API key created for user: {current_user.username}")

    rate_limits = get_rate_limits_for_tier(current_user.tier)

    return APIKeyResponse(
        key=api_key.key,
        tier=current_user.tier,
        rate_limit_per_minute=rate_limits["per_minute"],
        max_upload_mb=rate_limits["max_upload_mb"],
    )


@router.get("/me", response_model=UserResponse, summary="Get current user info")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get information about the currently authenticated user."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        tier=current_user.tier,
        created_at=current_user.created_at,
    )


@router.get("/rate-limits", response_model=RateLimitInfo, summary="Get rate limits")
async def get_rate_limits(current_user: User = Depends(get_current_user)):
    """Get rate limits for the current user's tier."""
    limits = get_rate_limits_for_tier(current_user.tier)
    return RateLimitInfo(
        tier=current_user.tier,
        per_minute=limits["per_minute"],
        per_hour=limits["per_hour"],
        per_day=limits["per_day"],
        max_upload_mb=limits["max_upload_mb"],
        max_frames=limits["max_frames"],
    )
