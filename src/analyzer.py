"""
Agentic Deepfake Analyzer
Main interface combining detection pipeline with agentic decision-making.

Author: Agentic Deepfake Classifier
"""

import logging
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from .detector import DetectionPipeline, VideoAnalysis
from .agents import DecisionAgent, CognitiveAgent, Verdict

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result from the agentic analyzer."""
    
    # Video info
    video_path: str
    duration_seconds: float
    
    # Verdict
    verdict: Verdict
    confidence: float
    
    # Scores
    average_fake_score: float
    max_fake_score: float
    min_fake_score: float
    
    # Frame stats
    frames_analyzed: int
    frames_with_faces: int
    
    # Human-readable outputs
    verdict_text: str
    explanation: str
    recommendation: str
    short_summary: str
    
    # Raw data for further analysis
    video_analysis: Optional[VideoAnalysis] = None
    
    def __str__(self) -> str:
        """String representation of analysis result."""
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
        """Convert to dictionary for JSON serialization."""
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


class DeepfakeAnalyzer:
    """
    Complete agentic deepfake analyzer.
    
    Combines:
    - Detection Pipeline (video → faces → classification)
    - Decision Agent (autonomous verdict determination)
    - Cognitive Agent (human-readable explanations)
    
    This is the main interface for analyzing videos.
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        sample_rate: float = 1.0,
        use_cuda: bool = False,
        max_frames: Optional[int] = None,
        fake_threshold: float = 0.7,
        suspicious_threshold: float = 0.4
    ):
        """
        Initialize the deepfake analyzer.
        
        Args:
            weights_path: Path to classifier weights
            sample_rate: Frame sampling rate (fps)
            use_cuda: Whether to use GPU
            max_frames: Maximum frames to analyze
            fake_threshold: Score threshold for FAKE verdict
            suspicious_threshold: Score threshold for SUSPICIOUS verdict
        """
        logger.info("Initializing Agentic Deepfake Analyzer...")
        
        # Initialize components
        self.pipeline = DetectionPipeline(
            weights_path=weights_path,
            sample_rate=sample_rate,
            use_cuda=use_cuda,
            max_frames=max_frames
        )
        
        self.decision_agent = DecisionAgent(
            fake_threshold=fake_threshold,
            suspicious_threshold=suspicious_threshold
        )
        
        self.cognitive_agent = CognitiveAgent()
        
        logger.info("Agentic Deepfake Analyzer ready!")
    
    def analyze(
        self, 
        video_path: str,
        show_progress: bool = True,
        include_raw_data: bool = False
    ) -> AnalysisResult:
        """
        Analyze a video for deepfakes.
        
        This is the main entry point for video analysis.
        
        Args:
            video_path: Path to the video file
            show_progress: Whether to show progress bar
            include_raw_data: Whether to include raw VideoAnalysis in result
            
        Returns:
            AnalysisResult with verdict, explanation, and recommendations
        """
        logger.info(f"Starting analysis: {video_path}")
        
        # Run detection pipeline
        video_analysis = self.pipeline.analyze_video(
            video_path, 
            show_progress=show_progress
        )
        
        # Get decision from agent
        decision = self.decision_agent.decide_from_analysis(video_analysis)
        
        # Generate cognitive response
        response = self.cognitive_agent.generate_response(decision, video_path)
        short_summary = self.cognitive_agent.generate_short_summary(decision)
        
        # Build result
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
            video_analysis=video_analysis if include_raw_data else None
        )
        
        logger.info(f"Analysis complete: {short_summary}")
        
        return result
    
    def quick_check(self, video_path: str, num_frames: int = 5) -> str:
        """
        Quick deepfake check with minimal frames.
        
        Args:
            video_path: Path to video
            num_frames: Number of frames to check (fewer = faster)
            
        Returns:
            Short summary string
        """
        logger.info(f"Quick check: {video_path} ({num_frames} frames)")
        
        # Use pipeline's quick check
        classification = self.pipeline.quick_check(video_path, num_frames)
        
        # Create minimal decision result for cognitive agent
        from .agents.decision_agent import DecisionResult
        
        decision = DecisionResult(
            verdict=Verdict.FAKE if classification.is_fake else Verdict.REAL,
            confidence=classification.confidence,
            average_fake_score=classification.fake_probability,
            frames_analyzed=num_frames,
            frames_with_faces=num_frames,
            score_variance=0.0,
            max_fake_score=classification.fake_probability,
            min_fake_score=classification.fake_probability
        )
        
        return self.cognitive_agent.generate_short_summary(decision)


def analyze_video(
    video_path: str,
    weights_path: Optional[str] = None,
    show_progress: bool = True
) -> AnalysisResult:
    """
    Convenience function for one-off video analysis.
    
    Args:
        video_path: Path to the video file
        weights_path: Path to model weights (optional)
        show_progress: Whether to show progress bar
        
    Returns:
        AnalysisResult with complete analysis
    """
    analyzer = DeepfakeAnalyzer(weights_path=weights_path)
    return analyzer.analyze(video_path, show_progress=show_progress)
