"""
API Schemas
Request/Response models for the API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


class VerdictEnum(str, Enum):
    """Verdict types."""

    REAL = "REAL"
    FAKE = "FAKE"
    SUSPICIOUS = "SUSPICIOUS"
    INCONCLUSIVE = "INCONCLUSIVE"


class JobStatusEnum(str, Enum):
    """Batch job status types."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TierEnum(str, Enum):
    """User tier types."""

    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request for video analysis (when using URL)."""

    video_url: str = Field(..., description="URL of the video to analyze")
    sample_rate: float = Field(1.0, ge=0.5, le=5.0, description="Frames per second")
    max_frames: int | None = Field(None, ge=1, le=100, description="Max frames")


class AnalyzeSettings(BaseModel):
    """Settings for video analysis."""

    sample_rate: float = Field(1.0, ge=0.5, le=5.0)
    max_frames: int | None = Field(None, ge=1, le=100)
    fake_threshold: float = Field(0.7, ge=0.5, le=0.95)
    suspicious_threshold: float = Field(0.4, ge=0.2, le=0.6)


class BatchAnalyzeRequest(BaseModel):
    """Request for batch video analysis."""

    include_quality_check: bool = Field(
        default=True, description="Include video quality assessment"
    )


class ComparativeAnalysisRequest(BaseModel):
    """Request for comparative analysis between two videos."""

    video1_description: str | None = Field(
        default="Original video", description="Description of first video"
    )
    video2_description: str | None = Field(
        default="Suspected deepfake", description="Description of second video"
    )
    include_differential: bool = Field(
        default=True, description="Include differential heatmap analysis"
    )


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.1.0"
    model_loaded: bool = True


class AnalysisResponse(BaseModel):
    """Analysis result response."""

    success: bool
    video_path: str
    duration_seconds: float
    verdict: VerdictEnum
    confidence: float = Field(..., ge=0.0, le=1.0)
    average_fake_score: float
    max_fake_score: float
    min_fake_score: float
    frames_analyzed: int
    frames_with_faces: int
    verdict_text: str
    explanation: str
    recommendation: str
    quality_metrics: dict | None = None
    frame_scores: list[float] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "video_path": "uploaded_video.mp4",
                "duration_seconds": 10.5,
                "verdict": "FAKE",
                "confidence": 0.85,
                "average_fake_score": 0.82,
                "max_fake_score": 0.91,
                "min_fake_score": 0.73,
                "frames_analyzed": 10,
                "frames_with_faces": 8,
                "verdict_text": "This video shows strong indicators of deepfake manipulation.",
                "explanation": "Analysis detected facial inconsistencies...",
                "recommendation": "Do not trust this video.",
            }
        }
    )


class QuickCheckResponse(BaseModel):
    """Quick check response."""

    success: bool
    summary: str
    verdict: VerdictEnum
    confidence: float


class QualityCheckResponse(BaseModel):
    """Video quality assessment response."""

    success: bool
    video_path: str
    overall_quality: float = Field(..., ge=0.0, le=1.0)
    resolution_score: float
    compression_score: float
    lighting_score: float
    face_clarity_score: float
    quality_label: str
    issues: list[str]
    recommendations: list[str]
    is_suitable: bool
    suitability_reason: str


class ComparativeAnalysisResponse(BaseModel):
    """Comparative analysis response."""

    success: bool
    video1_path: str
    video2_path: str
    video1_result: AnalysisResponse
    video2_result: AnalysisResponse
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    differential_analysis: str
    conclusion: str
    frame_scores_video1: list[float] | None = None
    frame_scores_video2: list[float] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchJobResponse(BaseModel):
    """Batch job creation response."""

    success: bool
    job_id: str
    status: JobStatusEnum
    total_videos: int
    message: str


class BatchJobStatusResponse(BaseModel):
    """Batch job status response."""

    job_id: str
    status: JobStatusEnum
    total_videos: int
    processed_videos: int
    failed_videos: int
    progress_percent: float
    results: list[AnalysisResponse] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = False
    error: str
    detail: str | None = None


class APIKeyResponse(BaseModel):
    """API key response."""

    key: str
    tier: TierEnum
    rate_limit_per_minute: int
    max_upload_mb: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    tier: TierEnum
    per_minute: int
    per_hour: int
    per_day: int
    max_upload_mb: int
    max_frames: int


class UserResponse(BaseModel):
    """User information response."""

    id: str
    username: str
    email: str | None = None
    tier: TierEnum
    created_at: datetime
