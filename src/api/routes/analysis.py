"""
Analysis Routes
Endpoints for video analysis with rate limiting and quality checks.
Uses shared classifier instance - model loaded once, shared by all requests.
"""

import os
import tempfile
import shutil
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from ..schemas import (
    AnalysisResponse,
    ErrorResponse,
    AnalyzeSettings,
    QualityCheckResponse,
    ComparativeAnalysisResponse,
    BatchJobResponse,
    BatchJobStatusResponse,
    JobStatusEnum,
)
from ..dependencies import get_classifier, get_current_user_from_auth
from ..security import get_rate_limits_for_tier, User
from src.detection import DeepfakeClassifier, VideoQualityAssessor
from src.pipeline import DeepfakeAnalyzer
from src.core.exceptions import VideoError, ClassifierError
from src.core import VideoQualityMetrics, get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])
security = HTTPBearer(auto_error=False)

# In-memory batch jobs store (replace with Redis/database in production)
batch_jobs: dict[str, dict] = {}


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


async def get_user_or_none(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User | None:
    """Get current user if authenticated, None otherwise."""
    return await get_current_user_from_auth(credentials)


@router.post(
    "",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Analyze a video for deepfakes",
    description="Upload a video file to analyze for deepfake manipulation.",
)
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze"),
    sample_rate: float = Form(1.0, ge=0.5, le=5.0),
    max_frames: Optional[int] = Form(None, ge=1),
    fake_threshold: float = Form(0.7, ge=0.5, le=0.95),
    suspicious_threshold: float = Form(0.4, ge=0.2, le=0.6),
    include_quality: bool = Form(True, description="Include quality assessment"),
    classifier: DeepfakeClassifier = Depends(get_classifier),
    current_user: User | None = Depends(get_user_or_none),
):
    """
    Analyze an uploaded video for deepfake manipulation.

    - **file**: Video file (MP4, AVI, MOV, MKV, WebM)
    - **sample_rate**: Frames per second to analyze (default: 1.0)
    - **max_frames**: Maximum frames to analyze (optional)
    - **fake_threshold**: Score threshold for FAKE verdict (default: 0.7)
    - **suspicious_threshold**: Score threshold for SUSPICIOUS verdict (default: 0.4)
    - **include_quality**: Include video quality assessment (default: true)
    """
    # Check file type
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

    # Check upload size limit based on user tier
    user_tier = current_user.tier if current_user else "free"
    rate_limits = get_rate_limits_for_tier(user_tier)
    max_size_bytes = rate_limits["max_upload_mb"] * 1024 * 1024

    # Get file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset

    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size for {user_tier} tier: {rate_limits['max_upload_mb']}MB",
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

        # Quality check first
        quality_metrics = None
        if include_quality:
            assessor = VideoQualityAssessor()
            quality = assessor.assess_video(temp_path)
            is_suitable, reason = assessor.is_suitable_for_analysis(quality)

            if not is_suitable:
                logger.warning(f"Video quality unsuitable: {reason}")

            quality_metrics = {
                "overall_quality": quality.overall_quality,
                "resolution_score": quality.resolution_score,
                "compression_score": quality.compression_score,
                "lighting_score": quality.lighting_score,
                "face_clarity_score": quality.face_clarity_score,
                "quality_label": assessor.get_quality_label(quality.overall_quality),
                "issues": quality.issues,
                "recommendations": quality.recommendations,
                "is_suitable": is_suitable,
                "suitability_reason": reason,
            }

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
            quality_metrics=quality_metrics,
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


@router.post(
    "/quality",
    response_model=QualityCheckResponse,
    summary="Check video quality",
    description="Assess video quality without running deepfake analysis.",
)
async def check_video_quality(
    file: UploadFile = File(..., description="Video file to assess"),
    classifier: DeepfakeClassifier = Depends(get_classifier),
):
    """
    Assess video quality for deepfake analysis suitability.

    Evaluates resolution, compression, lighting, and face clarity.
    """
    allowed_types = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "video/webm"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        assessor = VideoQualityAssessor()
        quality = assessor.assess_video(temp_path)
        is_suitable, reason = assessor.is_suitable_for_analysis(quality)
        quality_label = assessor.get_quality_label(quality.overall_quality)

        return QualityCheckResponse(
            success=True,
            video_path=file.filename,
            overall_quality=quality.overall_quality,
            resolution_score=quality.resolution_score,
            compression_score=quality.compression_score,
            lighting_score=quality.lighting_score,
            face_clarity_score=quality.face_clarity_score,
            quality_label=quality_label,
            issues=quality.issues,
            recommendations=quality.recommendations,
            is_suitable=is_suitable,
            suitability_reason=reason,
        )

    except Exception as e:
        logger.exception("Quality check failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post(
    "/compare",
    response_model=ComparativeAnalysisResponse,
    summary="Compare two videos",
    description="Compare original vs suspected deepfake side-by-side.",
)
async def compare_videos(
    video1: UploadFile = File(..., description="First video (e.g., original)"),
    video2: UploadFile = File(..., description="Second video (e.g., suspected deepfake)"),
    video1_description: str = Form("Original video"),
    video2_description: str = Form("Suspected deepfake"),
    classifier: DeepfakeClassifier = Depends(get_classifier),
):
    """
    Compare two videos side-by-side for deepfake analysis.

    Useful for comparing an original video against a suspected manipulation.
    """
    # Process both videos
    results = []
    temp_paths = []

    for idx, file in enumerate([video1, video2], 1):
        temp_path = None
        try:
            suffix = os.path.splitext(file.filename)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
            temp_paths.append(temp_path)

            settings = AnalyzeSettings()
            analyzer = create_analyzer_with_settings(classifier, settings)
            result = analyzer.analyze(temp_path, show_progress=False)

            results.append(
                AnalysisResponse(
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
            )

        except Exception as e:
            logger.exception(f"Analysis failed for video {idx}")
            raise HTTPException(status_code=500, detail=f"Error analyzing video {idx}: {str(e)}")

    # Calculate similarity score (simple difference-based metric)
    score_diff = abs(results[0].average_fake_score - results[1].average_fake_score)
    similarity_score = 1.0 - score_diff

    # Generate differential analysis
    if results[0].verdict == results[1].verdict:
        conclusion = f"Both videos show consistent results: {results[0].verdict}"
    else:
        conclusion = (
            f"Videos show different results. "
            f"{video1_description}: {results[0].verdict}, "
            f"{video2_description}: {results[1].verdict}"
        )

    differential = (
        f"Comparison between '{video1_description}' and '{video2_description}':\n\n"
        f"- Video 1 avg fake score: {results[0].average_fake_score:.1%}\n"
        f"- Video 2 avg fake score: {results[1].average_fake_score:.1%}\n"
        f"- Difference: {score_diff:.1%}\n"
        f"- Similarity: {similarity_score:.1%}\n\n"
        f"Verdict comparison:\n"
        f"- {video1_description}: {results[0].verdict} ({results[0].confidence:.0%} confidence)\n"
        f"- {video2_description}: {results[1].verdict} ({results[1].confidence:.0%} confidence)"
    )

    # Cleanup
    for temp_path in temp_paths:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    return ComparativeAnalysisResponse(
        success=True,
        video1_path=video1.filename,
        video2_path=video2.filename,
        video1_result=results[0],
        video2_result=results[1],
        similarity_score=similarity_score,
        differential_analysis=differential,
        conclusion=conclusion,
    )


@router.post(
    "/batch",
    response_model=BatchJobResponse,
    summary="Batch analyze multiple videos",
    description="Upload multiple videos for batch processing.",
)
async def batch_analyze(
    files: list[UploadFile] = File(..., description="Video files to analyze"),
    include_quality: bool = Form(True),
    background_tasks: BackgroundTasks = None,
    classifier: DeepfakeClassifier = Depends(get_classifier),
    current_user: User | None = Depends(get_user_or_none),
):
    """
    Analyze multiple videos in batch.

    Returns a job ID to track progress. Results available via /batch/{job_id}/status.
    """
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    # Check tier limits
    user_tier = current_user.tier if current_user else "free"
    rate_limits = get_rate_limits_for_tier(user_tier)

    if len(files) > 10 and user_tier == "free":
        raise HTTPException(
            status_code=403,
            detail=f"Free tier limited to 10 videos per batch. Current: {len(files)}",
        )

    # Create job
    job_id = str(uuid.uuid4())
    batch_jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatusEnum.PENDING,
        "total_videos": len(files),
        "processed_videos": 0,
        "failed_videos": 0,
        "results": [],
        "errors": [],
        "created_at": None,  # Will be set
        "completed_at": None,
    }

    # Start background processing
    if background_tasks:
        background_tasks.add_task(
            process_batch_job,
            job_id,
            files,
            include_quality,
            classifier,
        )

    return BatchJobResponse(
        success=True,
        job_id=job_id,
        status=JobStatusEnum.PENDING,
        total_videos=len(files),
        message=f"Batch job created. {len(files)} videos queued for processing.",
    )


@router.get(
    "/batch/{job_id}",
    response_model=BatchJobStatusResponse,
    summary="Get batch job status",
    description="Retrieve status and results of a batch processing job.",
)
async def get_batch_status(job_id: str):
    """Get status and results of a batch job."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = batch_jobs[job_id]
    progress = (job["processed_videos"] / job["total_videos"] * 100) if job["total_videos"] > 0 else 0

    return BatchJobStatusResponse(
        job_id=job_id,
        status=job["status"],
        total_videos=job["total_videos"],
        processed_videos=job["processed_videos"],
        failed_videos=job["failed_videos"],
        progress_percent=progress,
        results=job["results"],
        errors=job["errors"],
        created_at=job["created_at"],
        completed_at=job["completed_at"],
    )


async def process_batch_job(
    job_id: str,
    files: list[UploadFile],
    include_quality: bool,
    classifier: DeepfakeClassifier,
):
    """Process batch job in background."""
    from datetime import datetime

    job = batch_jobs[job_id]
    job["created_at"] = datetime.utcnow()
    job["status"] = JobStatusEnum.PROCESSING

    settings = AnalyzeSettings()
    analyzer = create_analyzer_with_settings(classifier, settings)

    for idx, file in enumerate(files):
        try:
            temp_path = None
            suffix = os.path.splitext(file.filename)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name

            result = analyzer.analyze(temp_path, show_progress=False)

            job["results"].append(
                AnalysisResponse(
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
            )
            job["processed_videos"] += 1

            if os.path.exists(temp_path):
                os.unlink(temp_path)

        except Exception as e:
            logger.exception(f"Batch error: {file.filename}")
            job["errors"].append(f"{file.filename}: {str(e)}")
            job["failed_videos"] += 1

    # Complete job
    job["status"] = JobStatusEnum.COMPLETED
    job["completed_at"] = datetime.utcnow()

    logger.info(f"Batch job {job_id} completed: {job['processed_videos']}/{job['total_videos']}")
