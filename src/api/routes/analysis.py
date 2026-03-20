"""
Analysis Routes
Endpoints for video analysis with rate limiting and quality checks.
Uses shared classifier instance - model loaded once, shared by all requests.
"""

from __future__ import annotations

import os
import tempfile
import shutil
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    BackgroundTasks,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

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
from ..dependencies import get_classifier, get_current_user_from_auth, get_job_store
from ..security import get_rate_limits_for_tier, User
from ...detection import DeepfakeClassifier, VideoQualityAssessor
from ...pipeline import DeepfakeAnalyzer
from ...core.exceptions import VideoError, ClassifierError
from ...core import get_logger
from ...workers.job_store import JobStore

logger = get_logger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])
security = HTTPBearer(auto_error=False)


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


def _extract_frame_scores(result) -> list[float]:
    """Pull per-frame fake scores from an AnalysisResult if available."""
    if result.video_analysis is None:
        return []
    return [
        f.classification.fake_probability
        for f in result.video_analysis.frame_analyses
        if f.face_detected and f.classification is not None
    ]


def _build_quality_metrics(assessor: VideoQualityAssessor, video_path: str) -> dict:
    """Run quality assessment and return metrics dict."""
    quality = assessor.assess_video(video_path)
    is_suitable, reason = assessor.is_suitable_for_analysis(quality)
    return {
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


def _result_to_response(
    result,
    filename: str,
    quality_metrics: dict | None = None,
    frame_scores: list[float] | None = None,
) -> AnalysisResponse:
    return AnalysisResponse(
        success=True,
        video_path=filename,
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
        frame_scores=frame_scores or [],
    )


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
    max_frames: int | None = Form(None, ge=1),
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

    user_tier = current_user.tier if current_user else "free"
    rate_limits = get_rate_limits_for_tier(user_tier)
    max_size_bytes = rate_limits["max_upload_mb"] * 1024 * 1024

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

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

        quality_metrics = None
        if include_quality:
            quality_metrics = _build_quality_metrics(VideoQualityAssessor(), temp_path)

        result = analyzer.analyze(temp_path, show_progress=False, include_raw_data=True)
        frame_scores = _extract_frame_scores(result)

        return _result_to_response(result, file.filename, quality_metrics, frame_scores)

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
    allowed_types = {
        "video/mp4",
        "video/avi",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
    }
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
    video2: UploadFile = File(
        ..., description="Second video (e.g., suspected deepfake)"
    ),
    video1_description: str = Form("Original video"),
    video2_description: str = Form("Suspected deepfake"),
    classifier: DeepfakeClassifier = Depends(get_classifier),
):
    """
    Compare two videos side-by-side for deepfake analysis.

    Returns per-frame fake score arrays (`frame_scores_video1`, `frame_scores_video2`)
    for client-side differential visualisation.
    """
    responses: list[AnalysisResponse] = []
    all_frame_scores: list[list[float]] = []
    temp_paths: list[str] = []

    for idx, (file, desc) in enumerate(
        [(video1, video1_description), (video2, video2_description)], 1
    ):
        temp_path = None
        try:
            suffix = os.path.splitext(file.filename)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
            temp_paths.append(temp_path)

            analyzer = create_analyzer_with_settings(classifier, AnalyzeSettings())
            result = analyzer.analyze(
                temp_path, show_progress=False, include_raw_data=True
            )
            frame_scores = _extract_frame_scores(result)
            all_frame_scores.append(frame_scores)
            responses.append(
                _result_to_response(result, file.filename, frame_scores=frame_scores)
            )

        except Exception as e:
            logger.exception(f"Analysis failed for video {idx}")
            raise HTTPException(
                status_code=500, detail=f"Error analyzing video {idx}: {str(e)}"
            )

    score_diff = abs(responses[0].average_fake_score - responses[1].average_fake_score)
    similarity_score = 1.0 - score_diff

    if responses[0].verdict == responses[1].verdict:
        conclusion = f"Both videos show consistent results: {responses[0].verdict}"
    else:
        conclusion = (
            f"Videos show different results. "
            f"{video1_description}: {responses[0].verdict}, "
            f"{video2_description}: {responses[1].verdict}"
        )

    differential = (
        f"Comparison between '{video1_description}' and '{video2_description}':\n\n"
        f"- Video 1 avg fake score: {responses[0].average_fake_score:.1%}\n"
        f"- Video 2 avg fake score: {responses[1].average_fake_score:.1%}\n"
        f"- Difference: {score_diff:.1%}\n"
        f"- Similarity: {similarity_score:.1%}\n\n"
        f"Verdict comparison:\n"
        f"- {video1_description}: {responses[0].verdict} ({responses[0].confidence:.0%} confidence)\n"
        f"- {video2_description}: {responses[1].verdict} ({responses[1].confidence:.0%} confidence)"
    )

    for p in temp_paths:
        if os.path.exists(p):
            os.unlink(p)

    return ComparativeAnalysisResponse(
        success=True,
        video1_path=video1.filename,
        video2_path=video2.filename,
        video1_result=responses[0],
        video2_result=responses[1],
        similarity_score=similarity_score,
        differential_analysis=differential,
        conclusion=conclusion,
        frame_scores_video1=all_frame_scores[0],
        frame_scores_video2=all_frame_scores[1],
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
    store: JobStore = Depends(get_job_store),
):
    """
    Analyze multiple videos in batch.

    Returns a job ID to track progress. Results available via /batch/{job_id}.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    user_tier = current_user.tier if current_user else "free"
    if len(files) > 10 and user_tier == "free":
        raise HTTPException(
            status_code=403,
            detail=f"Free tier limited to 10 videos per batch. Current: {len(files)}",
        )

    job_id = store.create(total_videos=len(files))

    if background_tasks:
        background_tasks.add_task(
            process_batch_job,
            job_id,
            files,
            include_quality,
            classifier,
            store,
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
async def get_batch_status(
    job_id: str,
    store: JobStore = Depends(get_job_store),
):
    """Get status and results of a batch job."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    total = job["total_videos"]
    processed = job["processed_videos"]
    progress = (processed / total * 100) if total > 0 else 0

    return BatchJobStatusResponse(
        job_id=job_id,
        status=job["status"],
        total_videos=total,
        processed_videos=processed,
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
    store: JobStore,
) -> None:
    """Process batch job in background."""
    from datetime import datetime

    store.update(
        job_id, {"status": "PROCESSING", "created_at": datetime.utcnow().isoformat()}
    )

    analyzer = create_analyzer_with_settings(classifier, AnalyzeSettings())
    results = []
    errors = []
    processed = 0
    failed = 0

    for file in files:
        temp_path = None
        try:
            suffix = os.path.splitext(file.filename)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name

            result = analyzer.analyze(
                temp_path, show_progress=False, include_raw_data=True
            )
            frame_scores = _extract_frame_scores(result)
            response = _result_to_response(
                result, file.filename, frame_scores=frame_scores
            )
            results.append(response.model_dump(mode="json"))
            processed += 1

            if os.path.exists(temp_path):
                os.unlink(temp_path)

        except Exception as e:
            logger.exception(f"Batch error: {file.filename}")
            errors.append(f"{file.filename}: {str(e)}")
            failed += 1

        store.update(
            job_id,
            {
                "processed_videos": processed,
                "failed_videos": failed,
                "results": results,
                "errors": errors,
            },
        )

    store.update(
        job_id,
        {
            "status": "COMPLETED",
            "completed_at": datetime.utcnow().isoformat(),
        },
    )
    logger.info(f"Batch job {job_id} completed: {processed}/{len(files)}")
