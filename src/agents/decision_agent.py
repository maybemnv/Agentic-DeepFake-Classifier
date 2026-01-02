"""
Decision Agent Module
Autonomous decision-making for deepfake detection verdicts.

Author: Agentic Deepfake Classifier
"""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


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


@dataclass
class DecisionResult:
    """Result from the decision agent."""
    verdict: Verdict
    confidence: float  # 0.0 to 1.0
    average_fake_score: float
    frames_analyzed: int
    frames_with_faces: int
    score_variance: float  # Consistency measure
    
    # Supporting evidence
    max_fake_score: float
    min_fake_score: float
    
    @property
    def confidence_percent(self) -> float:
        """Confidence as percentage."""
        return self.confidence * 100
    
    @property
    def is_high_confidence(self) -> bool:
        """Whether this is a high-confidence decision."""
        return self.confidence >= 0.8


class DecisionAgent:
    """
    Autonomous agent for making deepfake detection decisions.
    
    Responsibilities:
    - Aggregate frame-level predictions
    - Apply threshold-based decision logic
    - Provide confidence scoring
    - Handle edge cases (no faces, inconsistent results)
    """
    
    # Thresholds from POC specification
    FAKE_THRESHOLD = 0.7      # >= 0.7 → FAKE
    SUSPICIOUS_LOW = 0.4      # 0.4 to 0.7 → SUSPICIOUS
    REAL_THRESHOLD = 0.4      # < 0.4 → REAL
    
    # Minimum faces required for confident decision
    MIN_FACES_FOR_DECISION = 3
    
    # Variance threshold for consistency check
    HIGH_VARIANCE_THRESHOLD = 0.15
    
    def __init__(
        self,
        fake_threshold: float = 0.7,
        suspicious_threshold: float = 0.4,
        min_faces: int = 3
    ):
        """
        Initialize the decision agent.
        
        Args:
            fake_threshold: Score threshold for FAKE verdict
            suspicious_threshold: Score threshold for SUSPICIOUS verdict
            min_faces: Minimum faces required for confident decision
        """
        self.fake_threshold = fake_threshold
        self.suspicious_threshold = suspicious_threshold
        self.min_faces = min_faces
        
        logger.info(f"Decision Agent initialized with thresholds: "
                   f"FAKE>={fake_threshold}, SUSPICIOUS>={suspicious_threshold}")
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores for consistency check."""
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance
    
    def _determine_verdict(
        self, 
        avg_score: float, 
        variance: float,
        num_faces: int
    ) -> Verdict:
        """
        Determine verdict based on score and variance.
        
        Args:
            avg_score: Average fake probability
            variance: Score variance
            num_faces: Number of faces analyzed
            
        Returns:
            Verdict enum value
        """
        # Not enough data
        if num_faces < self.min_faces:
            if num_faces == 0:
                return Verdict.INCONCLUSIVE
            # Make decision but note lower confidence
            logger.warning(f"Low face count ({num_faces}), decision may be less reliable")
        
        # High variance indicates inconsistent detection
        if variance > self.HIGH_VARIANCE_THRESHOLD:
            logger.warning(f"High variance ({variance:.3f}), results may be inconsistent")
        
        # Apply thresholds
        if avg_score >= self.fake_threshold:
            return Verdict.FAKE
        elif avg_score >= self.suspicious_threshold:
            return Verdict.SUSPICIOUS
        else:
            return Verdict.REAL
    
    def _calculate_confidence(
        self, 
        avg_score: float, 
        variance: float,
        num_faces: int
    ) -> float:
        """
        Calculate confidence score for the decision.
        
        Args:
            avg_score: Average fake probability
            variance: Score variance
            num_faces: Number of faces analyzed
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from how far from decision boundary
        if avg_score >= 0.5:
            # For fake predictions, confidence increases with score
            base_confidence = avg_score
        else:
            # For real predictions, confidence increases as score approaches 0
            base_confidence = 1.0 - avg_score
        
        # Penalize for low sample size
        sample_factor = min(1.0, num_faces / self.min_faces)
        
        # Penalize for high variance (inconsistent predictions)
        variance_penalty = max(0.0, 1.0 - variance * 5)
        
        # Combine factors
        confidence = base_confidence * sample_factor * variance_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def decide(self, fake_scores: List[float]) -> DecisionResult:
        """
        Make a decision based on frame-level fake scores.
        
        Args:
            fake_scores: List of fake probability scores from classifier
            
        Returns:
            DecisionResult with verdict and supporting data
        """
        num_faces = len(fake_scores)
        
        # Handle empty case
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
        
        # Calculate statistics
        avg_score = sum(fake_scores) / num_faces
        variance = self._calculate_variance(fake_scores)
        max_score = max(fake_scores)
        min_score = min(fake_scores)
        
        # Determine verdict
        verdict = self._determine_verdict(avg_score, variance, num_faces)
        
        # Calculate confidence
        confidence = self._calculate_confidence(avg_score, variance, num_faces)
        
        logger.info(f"Decision: {verdict.value} (confidence: {confidence:.2%})")
        logger.debug(f"Stats: avg={avg_score:.3f}, var={variance:.3f}, "
                    f"range=[{min_score:.3f}, {max_score:.3f}]")
        
        return DecisionResult(
            verdict=verdict,
            confidence=confidence,
            average_fake_score=avg_score,
            frames_analyzed=num_faces,  # Will be updated by analyzer
            frames_with_faces=num_faces,
            score_variance=variance,
            max_fake_score=max_score,
            min_fake_score=min_score
        )
    
    def decide_from_analysis(self, video_analysis) -> DecisionResult:
        """
        Make a decision from a VideoAnalysis object.
        
        Args:
            video_analysis: VideoAnalysis from detection pipeline
            
        Returns:
            DecisionResult with verdict and supporting data
        """
        fake_scores = video_analysis.fake_scores
        result = self.decide(fake_scores)
        
        # Update frame counts from actual analysis
        result.frames_analyzed = video_analysis.total_frames_analyzed
        
        return result
