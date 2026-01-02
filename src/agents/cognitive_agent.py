"""
Cognitive Agent Module
Generates human-readable explanations for deepfake detection results.

Author: Agentic Deepfake Classifier
"""

from typing import Optional
from dataclasses import dataclass
import logging

from .decision_agent import DecisionResult, Verdict

logger = logging.getLogger(__name__)


@dataclass
class CognitiveResponse:
    """Human-readable response from cognitive agent."""
    verdict_text: str
    explanation: str
    technical_summary: str
    recommendation: str
    confidence_text: str


class CognitiveAgent:
    """
    Generates human-readable explanations for detection results.
    
    Responsibilities:
    - Convert numeric outputs to natural language
    - Provide contextual explanations
    - Generate actionable recommendations
    """
    
    # Explanation templates
    VERDICT_TEMPLATES = {
        Verdict.FAKE: {
            "high_confidence": "This video shows strong indicators of deepfake manipulation.",
            "medium_confidence": "This video likely contains deepfake manipulation, though some uncertainty remains.",
            "low_confidence": "This video may be manipulated, but the evidence is not conclusive."
        },
        Verdict.REAL: {
            "high_confidence": "This video appears to be authentic with high confidence.",
            "medium_confidence": "This video appears genuine, though minor inconsistencies were noted.",
            "low_confidence": "This video seems authentic, but could not be verified with high certainty."
        },
        Verdict.SUSPICIOUS: {
            "high_confidence": "This video contains suspicious elements that warrant further investigation.",
            "medium_confidence": "Some aspects of this video raise concerns but are not definitively problematic.",
            "low_confidence": "This video has some unusual characteristics, but they may be explained by natural factors."
        },
        Verdict.INCONCLUSIVE: {
            "default": "Unable to make a determination. No faces were detected or insufficient data was available."
        }
    }
    
    TECHNICAL_TEMPLATES = {
        "fake": "Analysis detected facial inconsistencies across {faces} frames. "
                "Average manipulation probability: {avg_score:.1%}. "
                "Score range: {min_score:.1%} - {max_score:.1%}.",
        
        "real": "Facial dynamics appear natural across {faces} analyzed frames. "
                "Average authenticity probability: {real_score:.1%}. "
                "Consistent results with low variance ({variance:.3f}).",
        
        "suspicious": "Mixed signals detected across {faces} frames. "
                      "Manipulation probability: {avg_score:.1%}. "
                      "High variance ({variance:.3f}) suggests inconsistent detection.",
        
        "inconclusive": "Analysis could not complete. {reason}"
    }
    
    RECOMMENDATIONS = {
        Verdict.FAKE: "Do not trust this video's authenticity. "
                      "Verify with the original source if possible. "
                      "Consider using additional forensic tools for confirmation.",
        
        Verdict.REAL: "This video can be considered authentic for general purposes. "
                      "For high-stakes decisions, consider additional verification.",
        
        Verdict.SUSPICIOUS: "Exercise caution with this video. "
                           "Seek the original source for verification. "
                           "Consider manual review by a forensic expert.",
        
        Verdict.INCONCLUSIVE: "Re-analyze with a longer video segment or different camera angle. "
                              "Ensure the video contains clear facial footage."
    }
    
    def __init__(self):
        """Initialize the cognitive agent."""
        logger.info("Cognitive Agent initialized")
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Map confidence score to level string."""
        if confidence >= 0.8:
            return "high_confidence"
        elif confidence >= 0.5:
            return "medium_confidence"
        else:
            return "low_confidence"
    
    def _get_confidence_text(self, confidence: float) -> str:
        """Generate confidence description text."""
        percent = confidence * 100
        
        if confidence >= 0.9:
            return f"Very High Confidence ({percent:.0f}%)"
        elif confidence >= 0.7:
            return f"High Confidence ({percent:.0f}%)"
        elif confidence >= 0.5:
            return f"Moderate Confidence ({percent:.0f}%)"
        elif confidence >= 0.3:
            return f"Low Confidence ({percent:.0f}%)"
        else:
            return f"Very Low Confidence ({percent:.0f}%)"
    
    def generate_response(
        self, 
        decision: DecisionResult,
        video_path: Optional[str] = None
    ) -> CognitiveResponse:
        """
        Generate a human-readable response for a detection decision.
        
        Args:
            decision: DecisionResult from decision agent
            video_path: Optional path to the analyzed video
            
        Returns:
            CognitiveResponse with explanation and recommendations
        """
        verdict = decision.verdict
        confidence_level = self._get_confidence_level(decision.confidence)
        
        # Get verdict text
        templates = self.VERDICT_TEMPLATES.get(verdict, {})
        verdict_text = templates.get(confidence_level, templates.get("default", "Analysis complete."))
        
        # Generate technical summary
        if verdict == Verdict.FAKE:
            technical_summary = self.TECHNICAL_TEMPLATES["fake"].format(
                faces=decision.frames_with_faces,
                avg_score=decision.average_fake_score,
                min_score=decision.min_fake_score,
                max_score=decision.max_fake_score
            )
        elif verdict == Verdict.REAL:
            technical_summary = self.TECHNICAL_TEMPLATES["real"].format(
                faces=decision.frames_with_faces,
                real_score=1.0 - decision.average_fake_score,
                variance=decision.score_variance
            )
        elif verdict == Verdict.SUSPICIOUS:
            technical_summary = self.TECHNICAL_TEMPLATES["suspicious"].format(
                faces=decision.frames_with_faces,
                avg_score=decision.average_fake_score,
                variance=decision.score_variance
            )
        else:
            reason = "No faces detected in the video." if decision.frames_with_faces == 0 else "Insufficient data."
            technical_summary = self.TECHNICAL_TEMPLATES["inconclusive"].format(reason=reason)
        
        # Get recommendation
        recommendation = self.RECOMMENDATIONS.get(verdict, "No specific recommendations.")
        
        # Generate confidence text
        confidence_text = self._get_confidence_text(decision.confidence)
        
        # Build full explanation
        explanation = self._build_explanation(decision, verdict_text, technical_summary)
        
        return CognitiveResponse(
            verdict_text=verdict_text,
            explanation=explanation,
            technical_summary=technical_summary,
            recommendation=recommendation,
            confidence_text=confidence_text
        )
    
    def _build_explanation(
        self, 
        decision: DecisionResult,
        verdict_text: str,
        technical_summary: str
    ) -> str:
        """Build a comprehensive explanation."""
        lines = [
            verdict_text,
            "",
            f"**Analysis Summary:**",
            f"- Frames analyzed: {decision.frames_analyzed}",
            f"- Faces detected: {decision.frames_with_faces}",
            f"- Detection confidence: {decision.confidence:.1%}",
            "",
            f"**Technical Details:**",
            technical_summary
        ]
        
        # Add variance warning if high
        if decision.score_variance > 0.1:
            lines.append("")
            lines.append("⚠️ Note: High variance in detection scores suggests "
                        "the video may have mixed authentic and manipulated segments.")
        
        return "\n".join(lines)
    
    def generate_short_summary(self, decision: DecisionResult) -> str:
        """
        Generate a one-line summary suitable for quick display.
        
        Args:
            decision: DecisionResult from decision agent
            
        Returns:
            Short summary string
        """
        verdict = decision.verdict
        confidence = decision.confidence
        
        return f"{verdict.emoji} {verdict.value}: {confidence:.0%} confident"
