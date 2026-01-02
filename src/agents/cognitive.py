"""
Cognitive Agent Module
Generates human-readable explanations for detection results.
"""

from typing import Optional
import logging

from ..core import Verdict, DecisionResult, CognitiveResponse

logger = logging.getLogger(__name__)


class CognitiveAgent:
    """
    Generates human-readable explanations for detection results.
    """
    
    VERDICT_TEMPLATES = {
        Verdict.FAKE: {
            "high": "This video shows strong indicators of deepfake manipulation.",
            "medium": "This video likely contains deepfake manipulation.",
            "low": "This video may be manipulated, but evidence is not conclusive."
        },
        Verdict.REAL: {
            "high": "This video appears to be authentic with high confidence.",
            "medium": "This video appears genuine with minor inconsistencies.",
            "low": "This video seems authentic but could not be verified with certainty."
        },
        Verdict.SUSPICIOUS: {
            "high": "This video contains suspicious elements warranting investigation.",
            "medium": "Some aspects of this video raise concerns.",
            "low": "This video has unusual characteristics."
        },
        Verdict.INCONCLUSIVE: {
            "default": "Unable to make a determination due to insufficient data."
        }
    }
    
    RECOMMENDATIONS = {
        Verdict.FAKE: "Do not trust this video. Verify with original source.",
        Verdict.REAL: "Video can be considered authentic for general purposes.",
        Verdict.SUSPICIOUS: "Exercise caution. Seek verification from original source.",
        Verdict.INCONCLUSIVE: "Re-analyze with better video quality or longer segment."
    }
    
    def __init__(self):
        logger.info("Cognitive Agent initialized")
    
    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        return "low"
    
    def _get_confidence_text(self, confidence: float) -> str:
        percent = confidence * 100
        if confidence >= 0.9:
            return f"Very High ({percent:.0f}%)"
        elif confidence >= 0.7:
            return f"High ({percent:.0f}%)"
        elif confidence >= 0.5:
            return f"Moderate ({percent:.0f}%)"
        return f"Low ({percent:.0f}%)"
    
    def generate_response(
        self, 
        decision: DecisionResult
    ) -> CognitiveResponse:
        """
        Generate a human-readable response.
        
        Args:
            decision: DecisionResult from decision agent
            
        Returns:
            CognitiveResponse
        """
        level = self._get_confidence_level(decision.confidence)
        templates = self.VERDICT_TEMPLATES.get(decision.verdict, {})
        verdict_text = templates.get(level, templates.get("default", "Analysis complete."))
        
        # Technical summary
        if decision.verdict == Verdict.FAKE:
            technical = (
                f"Detected facial inconsistencies across {decision.frames_with_faces} frames. "
                f"Average manipulation probability: {decision.average_fake_score:.1%}."
            )
        elif decision.verdict == Verdict.REAL:
            technical = (
                f"Facial dynamics appear natural across {decision.frames_with_faces} frames. "
                f"Authenticity probability: {1-decision.average_fake_score:.1%}."
            )
        else:
            technical = f"Analyzed {decision.frames_with_faces} frames with mixed results."
        
        explanation = (
            f"{verdict_text}\n\n"
            f"**Analysis Summary:**\n"
            f"- Frames analyzed: {decision.frames_analyzed}\n"
            f"- Faces detected: {decision.frames_with_faces}\n"
            f"- Detection confidence: {decision.confidence:.1%}\n\n"
            f"**Technical Details:**\n{technical}"
        )
        
        return CognitiveResponse(
            verdict_text=verdict_text,
            explanation=explanation,
            technical_summary=technical,
            recommendation=self.RECOMMENDATIONS.get(decision.verdict, ""),
            confidence_text=self._get_confidence_text(decision.confidence)
        )
    
    def generate_short_summary(self, decision: DecisionResult) -> str:
        """Generate a one-line summary."""
        return f"{decision.verdict.emoji} {decision.verdict.value}: {decision.confidence:.0%} confident"
