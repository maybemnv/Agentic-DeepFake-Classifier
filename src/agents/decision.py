"""
Decision Agent Module
Autonomous decision-making for deepfake detection verdicts.
"""

from typing import List
import logging

from ..core import Verdict, DecisionResult, DecisionConfig, DECISION_CONFIG

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    Autonomous agent for making deepfake detection decisions.
    """
    
    def __init__(self, config: DecisionConfig = None):
        """
        Initialize the decision agent.
        
        Args:
            config: Decision configuration
        """
        self.config = config or DECISION_CONFIG
        logger.info(f"Decision Agent initialized: FAKE>={self.config.fake_threshold}")
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores for consistency check."""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)
    
    def _determine_verdict(
        self, 
        avg_score: float, 
        variance: float,
        num_faces: int
    ) -> Verdict:
        """Determine verdict based on score and variance."""
        if num_faces < self.config.min_faces_for_decision:
            if num_faces == 0:
                return Verdict.INCONCLUSIVE
            logger.warning(f"Low face count ({num_faces})")
        
        if avg_score >= self.config.fake_threshold:
            return Verdict.FAKE
        elif avg_score >= self.config.suspicious_threshold:
            return Verdict.SUSPICIOUS
        else:
            return Verdict.REAL
    
    def _calculate_confidence(
        self, 
        avg_score: float, 
        variance: float,
        num_faces: int
    ) -> float:
        """Calculate confidence score for the decision."""
        base_confidence = avg_score if avg_score >= 0.5 else (1.0 - avg_score)
        sample_factor = min(1.0, num_faces / self.config.min_faces_for_decision)
        variance_penalty = max(0.0, 1.0 - variance * 5)
        confidence = base_confidence * sample_factor * variance_penalty
        return max(0.0, min(1.0, confidence))
    
    def decide(self, fake_scores: List[float]) -> DecisionResult:
        """
        Make a decision based on frame-level fake scores.
        
        Args:
            fake_scores: List of fake probability scores
            
        Returns:
            DecisionResult
        """
        num_faces = len(fake_scores)
        
        if num_faces == 0:
            return DecisionResult(
                verdict=Verdict.INCONCLUSIVE,
                confidence=0.0,
                average_fake_score=0.0,
                frames_analyzed=0,
                frames_with_faces=0,
                score_variance=0.0,
                max_fake_score=0.0,
                min_fake_score=0.0
            )
        
        avg_score = sum(fake_scores) / num_faces
        variance = self._calculate_variance(fake_scores)
        
        verdict = self._determine_verdict(avg_score, variance, num_faces)
        confidence = self._calculate_confidence(avg_score, variance, num_faces)
        
        logger.info(f"Decision: {verdict.value} (confidence: {confidence:.2%})")
        
        return DecisionResult(
            verdict=verdict,
            confidence=confidence,
            average_fake_score=avg_score,
            frames_analyzed=num_faces,
            frames_with_faces=num_faces,
            score_variance=variance,
            max_fake_score=max(fake_scores),
            min_fake_score=min(fake_scores)
        )
    
    def decide_from_analysis(self, video_analysis) -> DecisionResult:
        """Make a decision from a VideoAnalysis object."""
        result = self.decide(video_analysis.fake_scores)
        result.frames_analyzed = video_analysis.total_frames_analyzed
        return result
