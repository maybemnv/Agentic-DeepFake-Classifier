"""
Core Models Module
All dataclasses and type definitions for the project.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


# =============================================================================
# ENUMS
# =============================================================================

class Verdict(Enum):
    """Possible verdicts from the decision agent."""
    REAL = "REAL"
    FAKE = "FAKE"
    SUSPICIOUS = "SUSPICIOUS"
    INCONCLUSIVE = "INCONCLUSIVE"
    
    @property
    def color(self) -> str:
        """Get display color for verdict."""
        colors = {
            Verdict.REAL: "green",
            Verdict.FAKE: "red",
            Verdict.SUSPICIOUS: "yellow",
            Verdict.INCONCLUSIVE: "gray"
        }
        return colors.get(self, "gray")
    
    @property
    def emoji(self) -> str:
        """Get emoji for verdict."""
        emojis = {
            Verdict.REAL: "✅",
            Verdict.FAKE: "🚨",
            Verdict.SUSPICIOUS: "⚠️",
            Verdict.INCONCLUSIVE: "❓"
        }
        return emojis.get(self, "❓")


# =============================================================================
# VIDEO MODELS
# =============================================================================

@dataclass
class VideoMetadata:
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

@dataclass
class FaceResult:
    """Result of face detection for a single face."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    cropped_face: np.ndarray
    confidence: float = 1.0


# =============================================================================
# CLASSIFICATION MODELS
# =============================================================================

@dataclass
class ClassificationResult:
    """Result of deepfake classification."""
    prediction: str  # "REAL" or "FAKE"
    real_probability: float
    fake_probability: float
    confidence: float
    
    @property
    def is_fake(self) -> bool:
        return self.prediction == "FAKE"


# =============================================================================
# FRAME ANALYSIS MODELS
# =============================================================================

@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_index: int
    face_detected: bool
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    classification: Optional[ClassificationResult] = None


@dataclass
class VideoAnalysis:
    """Complete analysis result for a video."""
    video_path: str
    metadata: VideoMetadata
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    
    @property
    def frames_with_faces(self) -> List[FrameAnalysis]:
        return [f for f in self.frame_analyses if f.face_detected]
    
    @property
    def total_frames_analyzed(self) -> int:
        return len(self.frame_analyses)
    
    @property
    def frames_with_faces_count(self) -> int:
        return len(self.frames_with_faces)
    
    @property
    def fake_scores(self) -> List[float]:
        return [
            f.classification.fake_probability 
            for f in self.frames_with_faces 
            if f.classification
        ]
    
    @property
    def average_fake_score(self) -> float:
        scores = self.fake_scores
        return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# DECISION MODELS
# =============================================================================

@dataclass
class DecisionResult:
    """Result from the decision agent."""
    verdict: Verdict
    confidence: float
    average_fake_score: float
    frames_analyzed: int
    frames_with_faces: int
    score_variance: float
    max_fake_score: float
    min_fake_score: float
    
    @property
    def confidence_percent(self) -> float:
        return self.confidence * 100
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8


# =============================================================================
# COGNITIVE RESPONSE MODELS
# =============================================================================

@dataclass
class CognitiveResponse:
    """Human-readable response from cognitive agent."""
    verdict_text: str
    explanation: str
    technical_summary: str
    recommendation: str
    confidence_text: str


# =============================================================================
# FINAL ANALYSIS RESULT
# =============================================================================

@dataclass
class AnalysisResult:
    """Complete analysis result from the agentic analyzer."""
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
    video_analysis: Optional[VideoAnalysis] = None
    
    def __str__(self) -> str:
        from pathlib import Path
        return (
            f"\n{'='*60}\n"
            f"DEEPFAKE ANALYSIS RESULT\n"
            f"{'='*60}\n"
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
            f"{'='*60}\n"
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
            "short_summary": self.short_summary
        }
