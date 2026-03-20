"""
Job Store
Abstract job store with Redis-backed and in-memory implementations.
Falls back to in-memory when Redis is unavailable.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any

from ..core import get_logger

logger = get_logger(__name__)

_EXPIRY_SECONDS = 3600  # jobs expire after 1 hour


class JobStore(ABC):
    """Abstract job store interface."""

    @abstractmethod
    def create(self, total_videos: int) -> str:
        """Create a new job and return its ID."""

    @abstractmethod
    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return job dict or None if not found."""

    @abstractmethod
    def update(self, job_id: str, data: dict[str, Any]) -> None:
        """Merge data into the existing job record."""

    @abstractmethod
    def exists(self, job_id: str) -> bool:
        """Return True if the job exists."""


class InMemoryJobStore(JobStore):
    """Simple in-memory store — does not survive restarts."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def create(self, total_videos: int) -> str:
        job_id = str(uuid.uuid4())
        self._store[job_id] = {
            "job_id": job_id,
            "status": "PENDING",
            "total_videos": total_videos,
            "processed_videos": 0,
            "failed_videos": 0,
            "results": [],
            "errors": [],
            "created_at": None,
            "completed_at": None,
        }
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self._store.get(job_id)

    def update(self, job_id: str, data: dict[str, Any]) -> None:
        if job_id in self._store:
            self._store[job_id].update(data)

    def exists(self, job_id: str) -> bool:
        return job_id in self._store


class RedisJobStore(JobStore):
    """Redis-backed job store — survives restarts and scales horizontally."""

    def __init__(self, redis_url: str) -> None:
        import redis as redis_lib

        self._client = redis_lib.from_url(redis_url, decode_responses=True)
        # Verify connection on init
        self._client.ping()
        logger.info("RedisJobStore connected", extra={"url": redis_url})

    def _key(self, job_id: str) -> str:
        return f"deepfake:job:{job_id}"

    def create(self, total_videos: int) -> str:
        job_id = str(uuid.uuid4())
        data = {
            "job_id": job_id,
            "status": "PENDING",
            "total_videos": total_videos,
            "processed_videos": 0,
            "failed_videos": 0,
            "results": "[]",
            "errors": "[]",
            "created_at": "",
            "completed_at": "",
        }
        self._client.hset(self._key(job_id), mapping=data)
        self._client.expire(self._key(job_id), _EXPIRY_SECONDS)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        raw = self._client.hgetall(self._key(job_id))
        if not raw:
            return None
        return {
            "job_id": raw["job_id"],
            "status": raw["status"],
            "total_videos": int(raw["total_videos"]),
            "processed_videos": int(raw["processed_videos"]),
            "failed_videos": int(raw["failed_videos"]),
            "results": json.loads(raw.get("results", "[]")),
            "errors": json.loads(raw.get("errors", "[]")),
            "created_at": raw.get("created_at") or None,
            "completed_at": raw.get("completed_at") or None,
        }

    def update(self, job_id: str, data: dict[str, Any]) -> None:
        serialised: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                serialised[k] = json.dumps(v, default=str)
            elif v is None:
                serialised[k] = ""
            else:
                serialised[k] = str(v)
        if serialised:
            self._client.hset(self._key(job_id), mapping=serialised)
            self._client.expire(self._key(job_id), _EXPIRY_SECONDS)

    def exists(self, job_id: str) -> bool:
        return bool(self._client.exists(self._key(job_id)))


def create_job_store() -> JobStore:
    """
    Build the appropriate job store.
    Uses Redis when REDIS_HOST is configured and reachable, else falls back to in-memory.
    """
    try:
        from ..core.config import settings

        redis_url = (
            f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
            if settings.redis_password
            else f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
        )
        store = RedisJobStore(redis_url)
        logger.info("Using RedisJobStore for batch jobs")
        return store
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}), falling back to InMemoryJobStore")
        return InMemoryJobStore()
