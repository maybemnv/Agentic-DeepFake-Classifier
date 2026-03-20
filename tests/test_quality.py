"""
Test Quality Assessment
Tests for video quality assessment module.
"""

import pytest
import numpy as np
import cv2
from src.detection.quality import VideoQualityAssessor
from src.core import VideoQualityMetrics


class TestVideoQualityAssessor:
    """Test VideoQualityAssessor class."""

    @pytest.fixture
    def assessor(self):
        """Create assessor instance."""
        return VideoQualityAssessor()

    @pytest.fixture
    def sample_frame(self):
        """Create a sample test frame."""
        # Create a simple test image (gray gradient)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(256):
            frame[i, :] = [i, i, i]
        return frame

    def test_resolution_score_high(self, assessor):
        """Test resolution score for high resolution."""
        score = assessor._assess_resolution(1920, 1080)
        assert score == 1.0

    def test_resolution_score_low(self, assessor):
        """Test resolution score for low resolution."""
        score = assessor._assess_resolution(320, 240)
        assert score <= 1.0
        assert score >= 0.0

    def test_resolution_score_very_low(self, assessor):
        """Test resolution score for very low resolution."""
        score = assessor._assess_resolution(160, 120)
        assert score < 0.5

    def test_compression_score(self, assessor, sample_frame):
        """Test compression assessment."""
        score = assessor._assess_compression(sample_frame)
        assert 0.0 <= score <= 1.0

    def test_lighting_score_good(self, assessor):
        """Test lighting score for well-lit image."""
        # Create well-lit frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        score = assessor._assess_lighting(frame)
        assert score >= 0.5

    def test_lighting_score_dark(self, assessor):
        """Test lighting score for dark image."""
        # Create dark frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 20
        score = assessor._assess_lighting(frame)
        assert score < 0.5

    def test_lighting_score_bright(self, assessor):
        """Test lighting score for overexposed image."""
        # Create bright frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
        score = assessor._assess_lighting(frame)
        assert score < 0.5

    def test_clarity_score_sharp(self, assessor):
        """Test clarity score for sharp image."""
        # Create sharp image with edges
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (500, 400), (255, 255, 255), -1)
        score = assessor._assess_clarity(frame)
        assert score > 0.5

    def test_clarity_score_blur(self, assessor):
        """Test clarity score for blurry image."""
        # Create uniform (blurry) image
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        score = assessor._assess_clarity(frame)
        assert score < 0.5

    def test_quality_label(self, assessor):
        """Test quality label generation."""
        assert assessor.get_quality_label(0.9) == "Excellent"
        assert assessor.get_quality_label(0.7) == "Good"
        assert assessor.get_quality_label(0.5) == "Fair"
        assert assessor.get_quality_label(0.3) == "Poor"

    def test_is_suitable_for_analysis_good(self, assessor):
        """Test suitability check for good quality."""
        metrics = VideoQualityMetrics(
            resolution_score=0.9,
            compression_score=0.8,
            lighting_score=0.9,
            face_clarity_score=0.85,
            overall_quality=0.86,
        )
        is_suitable, reason = assessor.is_suitable_for_analysis(metrics)
        assert is_suitable is True

    def test_is_suitable_for_analysis_poor(self, assessor):
        """Test suitability check for poor quality."""
        metrics = VideoQualityMetrics(
            resolution_score=0.2,
            compression_score=0.3,
            lighting_score=0.2,
            face_clarity_score=0.25,
            overall_quality=0.24,
        )
        is_suitable, reason = assessor.is_suitable_for_analysis(metrics, min_score=0.4)
        assert is_suitable is False
        assert "quality too low" in reason.lower()
