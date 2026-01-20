"""
Analysis Routes
Endpoints for video analysis.
Uses shared classifier instance - model loaded once, shared by all requests.
"""

import os
import tempfile
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
import logging

from ..schemas import AnalysisResponse, ErrorResponse, AnalyzeSettings
from ..dependencies import get_classifier
from src.detection import DeepfakeClassifier
from src.pipeline import DeepfakeAnalyzer
from src.core.exceptions import VideoError, ClassifierError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])


def create_analyzer_with_settings(
        classifier: DeepfakeClassifier, settings: AnalyzeSettings
) -> DeepfakeAnalyzer:
    """Create analyzer with shared classifier and given settings."""
    return DeepfakeAnalyzer(
        classifier=classifier,
        sample_rate=settings.sample_rate,
        max_frames=settings.max_frames,
        fake_threshold=settings.fake_threshold,
        suspicious_threshold=settings.suspicious_threshold,
    )


@router.post(
    "",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Analyze a video for deepfakes",
    description="Upload a video file to analyze for deepfake manipulation. Model loaded once, shared by all requests.",
)
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze"),
    sample_rate: float = Form(1.0, ge=0.5, le=5.0),
    max_frames: Optional[int] = Form(None, ge=1, le=100),
    fake_threshold: float = Form(0.7, ge=0.5, le=0.95),
    suspicious_threshold: float = Form(0.4, ge=0.2, le=0.6),
    classifier: DeepfakeClassifier = Depends(get_classifier),
):
    """
    Analyze an uploaded video for deepfake manipulation.

    - **file**: Video file (MP4, AVI, MOV, MKV, WebM)
    - **sample_rate**: Frames per second to analyze (default: 1.0)
    - **max_frames**: Maximum frames to analyze (optional)
    - **fake_threshold**: Score threshold for FAKE verdict (default: 0.7)
    - **suspicious_threshold**: Score threshold for SUSPICIOUS verdict (default: 0.4)

    Uses shared classifier instance - model loaded on first request only.
    """
    allowed_types = {
        "video/mp4",
        "video/avi",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
    }
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: MP4, AVI, MOV, MKV, WebM",
        )

    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        settings = AnalyzeSettings(
            sample_rate=sample_rate,
            max_frames=max_frames,
            fake_threshold=fake_threshold,
            suspicious_threshold=suspicious_threshold,
        )

        analyzer = create_analyzer_with_settings(classifier, settings)

        result = analyzer.analyze(temp_path, show_progress=False)

        return AnalysisResponse(
            success=True,
            video_path=file.filename,
            duration_seconds=result.duration_seconds,
            verdict=result.verdict.value,
            confidence=result.confidence,
            average_fake_score=result.average_fake_score,
            max_fake_score=result.max_fake_score,
            min_fake_score=result.min_fake_score,
            frames_analyzed=result.frames_analyzed,
            frames_with_faces=result.frames_with_faces,
            verdict_text=result.verdict_text,
            explanation=result.explanation,
            recommendation=result.recommendation,
        )

    except VideoError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ClassifierError as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)



