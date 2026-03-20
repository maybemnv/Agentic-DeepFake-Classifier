"""
Test Core Models
Tests for Pydantic models and configuration.
"""

import pytest
from datetime import datetime
from src.core.models import (
    Verdict,
    VideoMetadata,
    ClassificationResult,
    FrameAnalysis,
    DecisionResult,
    AnalysisResult,
    VideoQualityMetrics,
)
from src.core.config import Settings


class TestVerdict:
    """Test Verdict enum."""

    def test_verdict_values(self):
        assert Verdict.REAL.value == "REAL"
        assert Verdict.FAKE.value == "FAKE"
        assert Verdict.SUSPICIOUS.value == "SUSPICIOUS"
        assert Verdict.INCONCLUSIVE.value == "INCONCLUSIVE"

    def test_verdict_emoji(self):
        assert Verdict.REAL.emoji == "✅"
        assert Verdict.FAKE.emoji == "🚨"
        assert Verdict.SUSPICIOUS.emoji == "⚠️"
        assert Verdict.INCONCLUSIVE.emoji == "❓"

    def test_verdict_color(self):
        assert Verdict.REAL.color == "green"
        assert Verdict.FAKE.color == "red"


class TestClassificationResult:
    """Test ClassificationResult model."""

    def test_valid_classification(self):
        result = ClassificationResult(
            prediction="FAKE",
            real_probability=0.3,
            fake_probability=0.7,
            confidence=0.7,
        )
        assert result.is_fake is True
        assert result.prediction == "FAKE"

    def test_real_classification(self):
        result = ClassificationResult(
            prediction="REAL",
            real_probability=0.8,
            fake_probability=0.2,
            confidence=0.8,
        )
        assert result.is_fake is False

    def test_invalid_prediction(self):
        with pytest.raises(ValueError):
            ClassificationResult(
                prediction="INVALID",
                real_probability=0.5,
                fake_probability=0.5,
                confidence=0.5,
            )

    def test_probability_bounds(self):
        with pytest.raises(ValueError):
            ClassificationResult(
                prediction="FAKE",
                real_probability=1.5,  # > 1.0
                fake_probability=0.7,
                confidence=0.7,
            )


class TestDecisionResult:
    """Test DecisionResult model."""

    def test_decision_result(self):
        result = DecisionResult(
            verdict=Verdict.FAKE,
            confidence=0.85,
            average_fake_score=0.82,
            frames_analyzed=10,
            frames_with_faces=8,
            score_variance=0.05,
            max_fake_score=0.91,
            min_fake_score=0.73,
        )
        assert result.confidence_percent == 85.0
        assert result.is_high_confidence is True

    def test_low_confidence(self):
        result = DecisionResult(
            verdict=Verdict.SUSPICIOUS,
            confidence=0.45,
            average_fake_score=0.55,
            frames_analyzed=5,
            frames_with_faces=3,
            score_variance=0.1,
            max_fake_score=0.65,
            min_fake_score=0.45,
        )
        assert result.is_high_confidence is False


class TestAnalysisResult:
    """Test AnalysisResult model."""

    def test_analysis_result_creation(self):
        result = AnalysisResult(
            video_path="/path/to/video.mp4",
            duration_seconds=30.5,
            verdict=Verdict.FAKE,
            confidence=0.85,
            average_fake_score=0.82,
            max_fake_score=0.91,
            min_fake_score=0.73,
            frames_analyzed=10,
            frames_with_faces=8,
            verdict_text="This video shows manipulation.",
            explanation="Detailed explanation...",
            recommendation="Do not trust.",
            short_summary="FAKE: 85% confident",
        )
        assert result.verdict == Verdict.FAKE
        assert isinstance(result.timestamp, datetime)

    def test_to_dict(self):
        result = AnalysisResult(
            video_path="/path/to/video.mp4",
            duration_seconds=30.5,
            verdict=Verdict.REAL,
            confidence=0.9,
            average_fake_score=0.15,
            max_fake_score=0.2,
            min_fake_score=0.1,
            frames_analyzed=10,
            frames_with_faces=8,
            verdict_text="This video appears authentic.",
            explanation="Detailed explanation...",
            recommendation="Can be trusted.",
            short_summary="REAL: 90% confident",
        )
        d = result.to_dict()
        assert d["verdict"] == "REAL"
        assert "timestamp" in d


class TestVideoQualityMetrics:
    """Test VideoQualityMetrics model."""

    def test_quality_metrics(self):
        metrics = VideoQualityMetrics(
            resolution_score=0.8,
            compression_score=0.7,
            lighting_score=0.9,
            face_clarity_score=0.85,
            overall_quality=0.81,
            issues=["Low resolution"],
            recommendations=["Use higher resolution"],
        )
        assert metrics.overall_quality == 0.81
        assert len(metrics.issues) == 1


class TestSettings:
    """Test Settings configuration."""

    def test_default_settings(self):
        settings = Settings()
        assert settings.app_name == "Agentic DeepFake Classifier"
        assert settings.api_port == 8000
        assert settings.fake_threshold == 0.7

    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("FAKE_THRESHOLD", "0.8")

        settings = Settings()
        assert settings.api_port == 9000
        assert settings.fake_threshold == 0.8
