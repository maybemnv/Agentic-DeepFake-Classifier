"""
Analysis Pipeline Module
Complete agentic analysis combining detection and decision-making.
Receives classifier as dependency - does not create it.
"""

from typing import Optional
from pathlib import Path
import logging

from ..core import AnalysisResult, Verdict
from ..agents import DecisionAgent, CognitiveAgent
from ..detection import DeepfakeClassifier
from .detection import DetectionPipeline

logger = logging.getLogger(__name__)


class DeepfakeAnalyzer:
    """
    Complete agentic deepfake analyzer.

    Combines:
    - Detection Pipeline (video → faces → classification)
    - Decision Agent (autonomous verdict)
    - Cognitive Agent (human explanations)

    Receives classifier as dependency - classifier lifecycle is managed externally.
    """

    def __init__(
        self,
        classifier: DeepfakeClassifier,
        sample_rate: float = 1.0,
        max_frames: Optional[int] = None,
        fake_threshold: float = 0.7,
        suspicious_threshold: float = 0.4,
    ):
        """
        Initialize the deepfake analyzer.

        Args:
            classifier: Pre-initialized DeepfakeClassifier instance
            sample_rate: Frame sampling rate (fps)
            max_frames: Maximum frames to analyze
            fake_threshold: Score threshold for FAKE verdict
            suspicious_threshold: Score threshold for SUSPICIOUS verdict
        """
        logger.info("Initializing Agentic Deepfake Analyzer...")

        self.pipeline = DetectionPipeline(
            classifier=classifier, sample_rate=sample_rate, max_frames=max_frames
        )

        from ..core import DecisionConfig

        decision_config = DecisionConfig(
            fake_threshold=fake_threshold, suspicious_threshold=suspicious_threshold
        )

        self.decision_agent = DecisionAgent(config=decision_config)
        self.cognitive_agent = CognitiveAgent()

        logger.info("Agentic Deepfake Analyzer ready!")

    def analyze(
        self,
        video_path: str,
        show_progress: bool = True,
        include_raw_data: bool = False,
    ) -> AnalysisResult:
        """
        Analyze a video for deepfakes.

        Args:
            video_path: Path to the video file
            show_progress: Whether to show progress bar
            include_raw_data: Whether to include raw VideoAnalysis

        Returns:
            AnalysisResult
        """
        logger.info(f"Starting analysis: {video_path}")

        # Run detection
        video_analysis = self.pipeline.analyze_video(video_path, show_progress)

        # Get decision
        decision = self.decision_agent.decide_from_analysis(video_analysis)

        # Generate explanation
        response = self.cognitive_agent.generate_response(decision)
        short_summary = self.cognitive_agent.generate_short_summary(decision)

        result = AnalysisResult(
            video_path=video_path,
            duration_seconds=video_analysis.metadata.duration_seconds,
            verdict=decision.verdict,
            confidence=decision.confidence,
            average_fake_score=decision.average_fake_score,
            max_fake_score=decision.max_fake_score,
            min_fake_score=decision.min_fake_score,
            frames_analyzed=decision.frames_analyzed,
            frames_with_faces=decision.frames_with_faces,
            verdict_text=response.verdict_text,
            explanation=response.explanation,
            recommendation=response.recommendation,
            short_summary=short_summary,
            video_analysis=video_analysis if include_raw_data else None,
        )

        logger.info(f"Analysis complete: {short_summary}")
        return result

    def quick_check(self, video_path: str, num_frames: int = 5) -> str:
        """Quick deepfake check with minimal frames."""
        old_max = self.pipeline.max_frames
        self.pipeline.max_frames = num_frames

        analysis = self.pipeline.analyze_video(video_path, show_progress=False)

        self.pipeline.max_frames = old_max

        decision = self.decision_agent.decide(analysis.fake_scores)
        return self.cognitive_agent.generate_short_summary(decision)


def analyze_video(
    video_path: str,
    weights_path: Optional[str] = None,
    sample_rate: float = 1.0,
    max_frames: Optional[int] = None,
    show_progress: bool = True,
) -> AnalysisResult:
    """
    Convenience function for one-off video analysis.

    Creates and manages its own classifier instance.
    For server use, create DeepfakeAnalyzer explicitly with a shared classifier.
    """
    from ..detection import DeepfakeClassifier

    classifier = DeepfakeClassifier(weights_path=weights_path)
    analyzer = DeepfakeAnalyzer(
        classifier=classifier, sample_rate=sample_rate, max_frames=max_frames
    )
    return analyzer.analyze(video_path, show_progress=show_progress)
