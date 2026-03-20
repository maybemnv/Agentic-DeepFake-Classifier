"""
Core Models Module
All Pydantic models and type definitions for the project.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict
from enum import Enum
from datetime import datetime
import numpy as np


# =============================================================================
# ENUMS
# =============================================================================


class Verdict(str, Enum):
    """Possible verdicts from the decision agent."""

    REAL = "REAL"
    FAKE = "FAKE"
    SUSPICIOUS = "SUSPICIOUS"
    INCONCLUSIVE = "INCONCLUSIVE"

    @property
    def color(self) -> str:
        """Get display color for verdict."""
        colors: dict[Verdict, str] = {
            Verdict.REAL: "green",
            Verdict.FAKE: "red",
            Verdict.SUSPICIOUS: "yellow",
            Verdict.INCONCLUSIVE: "gray",
        }
        return colors.get(self, "gray")

    @property
    def emoji(self) -> str:
        """Get emoji for verdict."""
        emojis: dict[Verdict, str] = {
            Verdict.REAL: "✅",
            Verdict.FAKE: "🚨",
            Verdict.SUSPICIOUS: "⚠️",
            Verdict.INCONCLUSIVE: "❓",
        }
        return emojis.get(self, "❓")


# =============================================================================
# VIDEO MODELS
# =============================================================================


class VideoMetadata(BaseModel):
    """Metadata about the processed video."""

    path: str
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
    format: str


# =============================================================================
# FACE DETECTION MODELS
# =============================================================================


class FaceResult(BaseModel):
    """Result of face detection for a single face."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    cropped_face: np.ndarray
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# =============================================================================
# CLASSIFICATION MODELS
# =============================================================================


class ClassificationResult(BaseModel):
    """Result of deepfake classification."""

    prediction: str  # "REAL" or "FAKE"
    real_probability: float = Field(ge=0.0, le=1.0)
    fake_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("prediction")
    @classmethod
    def validate_prediction(cls, v: str) -> str:
        if v not in ("REAL", "FAKE"):
            raise ValueError("Prediction must be 'REAL' or 'FAKE'")
        return v

    @computed_field
    @property
    def is_fake(self) -> bool:
        return self.prediction == "FAKE"


# =============================================================================
# FRAME ANALYSIS MODELS
# =============================================================================


class FrameAnalysis(BaseModel):
    """Analysis result for a single frame."""

    frame_index: int
    face_detected: bool
    face_bbox: tuple[int, int, int, int] | None = None
    classification: ClassificationResult | None = None


class VideoAnalysis(BaseModel):
    """Complete analysis result for a video."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_path: str
    metadata: VideoMetadata
    frame_analyses: list[FrameAnalysis] = Field(default_factory=list)

    @computed_field
    @property
    def frames_with_faces(self) -> list[FrameAnalysis]:
        return [f for f in self.frame_analyses if f.face_detected]

    @computed_field
    @property
    def total_frames_analyzed(self) -> int:
        return len(self.frame_analyses)

    @computed_field
    @property
    def frames_with_faces_count(self) -> int:
        return len(self.frames_with_faces)

    @computed_field
    @property
    def fake_scores(self) -> list[float]:
        return [
            f.classification.fake_probability for f in self.frames_with_faces if f.classification
        ]

    @computed_field
    @property
    def average_fake_score(self) -> float:
        scores = self.fake_scores
        return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# DECISION MODELS
# =============================================================================


class DecisionResult(BaseModel):
    """Result from the decision agent."""

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    average_fake_score: float = Field(ge=0.0, le=1.0)
    frames_analyzed: int = Field(ge=0)
    frames_with_faces: int = Field(ge=0)
    score_variance: float = Field(ge=0.0)
    max_fake_score: float = Field(ge=0.0, le=1.0)
    min_fake_score: float = Field(ge=0.0, le=1.0)

    @computed_field
    @property
    def confidence_percent(self) -> float:
        return self.confidence * 100

    @computed_field
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8


# =============================================================================
# COGNITIVE RESPONSE MODELS
# =============================================================================


class CognitiveResponse(BaseModel):
    """Human-readable response from cognitive agent."""

    verdict_text: str
    explanation: str
    technical_summary: str
    recommendation: str
    confidence_text: str


# =============================================================================
# FINAL ANALYSIS RESULT
# =============================================================================


class AnalysisResult(BaseModel):
    """Complete analysis result from the agentic analyzer."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_serializers={
            Verdict: lambda v: v.value,
            datetime: lambda dt: dt.isoformat(),
        },
    )

    video_path: str
    duration_seconds: float
    verdict: Verdict
    confidence: float
    average_fake_score: float
    max_fake_score: float
    min_fake_score: float
    frames_analyzed: int
    frames_with_faces: int
    verdict_text: str
    explanation: str
    recommendation: str
    short_summary: str
    video_analysis: VideoAnalysis | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        from pathlib import Path

        return (
            f"\n{'=' * 60}\n"
            f"DEEPFAKE ANALYSIS RESULT\n"
            f"{'=' * 60}\n"
            f"\n📁 Video: {Path(self.video_path).name}\n"
            f"⏱️  Duration: {self.duration_seconds:.1f}s\n"
            f"\n{self.verdict.emoji} VERDICT: {self.verdict.value}\n"
            f"📊 Confidence: {self.confidence:.1%}\n"
            f"\n--- Explanation ---\n"
            f"{self.verdict_text}\n"
            f"\n--- Technical Summary ---\n"
            f"• Frames analyzed: {self.frames_analyzed}\n"
            f"• Faces detected: {self.frames_with_faces}\n"
            f"• Avg fake score: {self.average_fake_score:.1%}\n"
            f"• Score range: {self.min_fake_score:.1%} - {self.max_fake_score:.1%}\n"
            f"\n--- Recommendation ---\n"
            f"{self.recommendation}\n"
            f"{'=' * 60}\n"
        )

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "duration_seconds": self.duration_seconds,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "average_fake_score": self.average_fake_score,
            "max_fake_score": self.max_fake_score,
            "min_fake_score": self.min_fake_score,
            "frames_analyzed": self.frames_analyzed,
            "frames_with_faces": self.frames_with_faces,
            "verdict_text": self.verdict_text,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "short_summary": self.short_summary,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# QUALITY ASSESSMENT MODELS
# =============================================================================


class VideoQualityMetrics(BaseModel):
    """Video quality assessment metrics."""

    resolution_score: float = Field(ge=0.0, le=1.0)
    compression_score: float = Field(ge=0.0, le=1.0)
    lighting_score: float = Field(ge=0.0, le=1.0)
    face_clarity_score: float = Field(ge=0.0, le=1.0)
    overall_quality: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# =============================================================================
# BATCH PROCESSING MODELS
# =============================================================================


class BatchJobStatus(str, Enum):
    """Status of a batch processing job."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BatchJobInfo(BaseModel):
    """Information about a batch processing job."""

    job_id: str
    status: BatchJobStatus
    total_videos: int
    processed_videos: int
    failed_videos: int
    created_at: datetime
    completed_at: datetime | None = None
    results: list[AnalysisResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# =============================================================================
# COMPARATIVE ANALYSIS MODELS
# =============================================================================


class ComparativeAnalysisResult(BaseModel):
    """Result of comparative analysis between two videos."""

    video1_path: str
    video2_path: str
    video1_result: AnalysisResult
    video2_result: AnalysisResult
    similarity_score: float = Field(ge=0.0, le=1.0)
    differential_analysis: str
    conclusion: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
