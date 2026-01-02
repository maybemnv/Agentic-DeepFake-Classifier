"""
API Schemas
Request/Response models for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class VerdictEnum(str, Enum):
    """Verdict types."""
    REAL = "REAL"
    FAKE = "FAKE"
    SUSPICIOUS = "SUSPICIOUS"
    INCONCLUSIVE = "INCONCLUSIVE"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request for video analysis (when using URL)."""
    video_url: str = Field(..., description="URL of the video to analyze")
    sample_rate: float = Field(1.0, ge=0.5, le=5.0, description="Frames per second")
    max_frames: Optional[int] = Field(None, ge=1, le=100, description="Max frames")


class AnalyzeSettings(BaseModel):
    """Settings for video analysis."""
    sample_rate: float = Field(1.0, ge=0.5, le=5.0)
    max_frames: Optional[int] = Field(None, ge=1, le=100)
    fake_threshold: float = Field(0.7, ge=0.5, le=0.95)
    suspicious_threshold: float = Field(0.4, ge=0.2, le=0.6)


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"


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
    
    class Config:
        json_schema_extra = {
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
                "recommendation": "Do not trust this video."
            }
        }


class QuickCheckResponse(BaseModel):
    """Quick check response."""
    success: bool
    summary: str
    verdict: VerdictEnum
    confidence: float


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
