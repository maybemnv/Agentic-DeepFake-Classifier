"""
Video Quality Assessment Module
Analyzes video quality before deepfake detection.
"""

import cv2
import numpy as np
from typing import Literal
import logging

from ..core import VideoQualityMetrics

logger = logging.getLogger(__name__)


class VideoQualityAssessor:
    """
    Assesses video quality for deepfake analysis suitability.

    Evaluates:
    - Resolution quality
    - Compression artifacts
    - Lighting conditions
    - Face clarity
    """

    # Quality thresholds
    MIN_RESOLUTION = (320, 240)
    RECOMMENDED_RESOLUTION = (640, 480)
    BLUR_THRESHOLD = 100
    LOW_LIGHT_THRESHOLD = 30
    HIGH_LIGHT_THRESHOLD = 220

    def __init__(self):
        """Initialize the quality assessor."""
        pass

    def assess_video(self, video_path: str, sample_frames: int = 10) -> VideoQualityMetrics:
        """
        Assess overall video quality.

        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample

        Returns:
            VideoQualityMetrics with scores and recommendations
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames == 0:
            cap.release()
            raise ValueError("Video has no frames")

        # Sample frame indices
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

        resolution_scores = []
        compression_scores = []
        lighting_scores = []
        clarity_scores = []

        issues = set()
        recommendations = set()

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            # Resolution score
            res_score = self._assess_resolution(width, height)
            resolution_scores.append(res_score)

            if res_score < 0.5:
                issues.add("Low resolution")
                recommendations.add("Use higher resolution video (minimum 640x480)")

            # Compression score
            comp_score = self._assess_compression(frame)
            compression_scores.append(comp_score)

            if comp_score < 0.5:
                issues.add("High compression artifacts")
                recommendations.add("Use less compressed video source")

            # Lighting score
            light_score = self._assess_lighting(frame)
            lighting_scores.append(light_score)

            if light_score < 0.5:
                issues.add("Poor lighting conditions")
                recommendations.add("Ensure adequate, even lighting")

            # Clarity/blur score
            clarity_score = self._assess_clarity(frame)
            clarity_scores.append(clarity_score)

            if clarity_score < 0.5:
                issues.add("Motion blur or out-of-focus")
                recommendations.add("Use stable, focused video")

        cap.release()

        # Calculate overall scores
        metrics = VideoQualityMetrics(
            resolution_score=float(np.mean(resolution_scores)) if resolution_scores else 0.0,
            compression_score=float(np.mean(compression_scores)) if compression_scores else 0.0,
            lighting_score=float(np.mean(lighting_scores)) if lighting_scores else 0.0,
            face_clarity_score=float(np.mean(clarity_scores)) if clarity_scores else 0.0,
            overall_quality=float(
                np.mean(
                    [
                        np.mean(resolution_scores) if resolution_scores else 0,
                        np.mean(compression_scores) if compression_scores else 0,
                        np.mean(lighting_scores) if lighting_scores else 0,
                        np.mean(clarity_scores) if clarity_scores else 0,
                    ]
                )
            ),
            issues=list(issues),
            recommendations=list(recommendations),
        )

        logger.info(
            f"Video quality assessed: {metrics.overall_quality:.2f}",
            extra={"video_path": video_path, "quality": metrics.overall_quality},
        )

        return metrics

    def _assess_resolution(self, width: int, height: int) -> float:
        """
        Assess resolution quality.

        Returns:
            Score from 0.0 to 1.0
        """
        min_w, min_h = self.MIN_RESOLUTION
        rec_w, rec_h = self.RECOMMENDED_RESOLUTION

        if width >= rec_w and height >= rec_h:
            return 1.0
        elif width >= min_w and height >= min_h:
            # Linear interpolation
            w_score = (width - min_w) / (rec_w - min_w)
            h_score = (height - min_h) / (rec_h - min_h)
            return float((w_score + h_score) / 2)
        else:
            # Below minimum
            w_score = max(0, width / min_w)
            h_score = max(0, height / min_h)
            return float((w_score + h_score) / 2) * 0.5

    def _assess_compression(self, frame: np.ndarray) -> float:
        """
        Assess compression artifacts using edge analysis.

        High compression creates blocky artifacts.

        Returns:
            Score from 0.0 to 1.0
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance (measures sharpness/artifacts)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize to 0-1 score
        # Higher variance = less compression (generally)
        if variance > 500:
            return 1.0
        elif variance > 100:
            return float((variance - 100) / 400)
        else:
            return float(variance / 100) * 0.5

    def _assess_lighting(self, frame: np.ndarray) -> float:
        """
        Assess lighting conditions.

        Returns:
            Score from 0.0 to 1.0
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Check if too dark or too bright
        if mean_brightness < self.LOW_LIGHT_THRESHOLD:
            # Too dark
            return float(mean_brightness / self.LOW_LIGHT_THRESHOLD) * 0.5
        elif mean_brightness > self.HIGH_LIGHT_THRESHOLD:
            # Too bright (washed out)
            return float((255 - mean_brightness) / (255 - self.HIGH_LIGHT_THRESHOLD)) * 0.5
        else:
            # Good brightness range
            brightness_score = 1.0

            # Also check contrast (std dev)
            if std_brightness > 50:
                contrast_score = 1.0
            elif std_brightness > 20:
                contrast_score = float(std_brightness / 50)
            else:
                contrast_score = float(std_brightness / 20) * 0.5

            return float((brightness_score + contrast_score) / 2)

    def _assess_clarity(self, frame: np.ndarray) -> float:
        """
        Assess image clarity (blur detection).

        Uses Laplacian variance method.

        Returns:
            Score from 0.0 to 1.0
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize to score
        if variance > self.BLUR_THRESHOLD * 2:
            return 1.0
        elif variance > self.BLUR_THRESHOLD:
            return float((variance - self.BLUR_THRESHOLD) / self.BLUR_THRESHOLD)
        else:
            return float(variance / self.BLUR_THRESHOLD) * 0.5

    def is_suitable_for_analysis(
        self, metrics: VideoQualityMetrics, min_score: float = 0.4
    ) -> tuple[bool, str]:
        """
        Check if video quality is suitable for deepfake analysis.

        Args:
            metrics: Quality metrics
            min_score: Minimum overall quality score

        Returns:
            Tuple of (is_suitable, reason)
        """
        if metrics.overall_quality < min_score:
            return (
                False,
                f"Video quality too low ({metrics.overall_quality:.2f} < {min_score}). "
                f"Issues: {', '.join(metrics.issues)}",
            )

        if metrics.resolution_score < 0.3:
            return False, "Resolution too low for reliable face detection"

        if metrics.lighting_score < 0.3:
            return False, "Lighting conditions too poor for analysis"

        return True, "Video quality is suitable for analysis"

    def get_quality_label(self, score: float) -> Literal["Poor", "Fair", "Good", "Excellent"]:
        """Get human-readable quality label."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
