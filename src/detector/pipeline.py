"""
Detection Pipeline Module
Orchestrates video processing, face detection, and classification.

Author: Agentic Deepfake Classifier
"""

import logging
from typing import List, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from .video_processor import VideoProcessor, VideoMetadata
from .face_detector import FaceDetector, FaceResult
from .classifier import DeepfakeClassifier, ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_index: int
    face_detected: bool
    face_bbox: Optional[tuple] = None
    classification: Optional[ClassificationResult] = None


@dataclass
class VideoAnalysis:
    """Complete analysis result for a video."""
    video_path: str
    metadata: VideoMetadata
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    
    @property
    def frames_with_faces(self) -> List[FrameAnalysis]:
        """Get only frames where faces were detected."""
        return [f for f in self.frame_analyses if f.face_detected]
    
    @property
    def total_frames_analyzed(self) -> int:
        """Total number of frames analyzed."""
        return len(self.frame_analyses)
    
    @property
    def frames_with_faces_count(self) -> int:
        """Number of frames with detected faces."""
        return len(self.frames_with_faces)
    
    @property
    def fake_scores(self) -> List[float]:
        """Get all fake probability scores."""
        return [
            f.classification.fake_probability 
            for f in self.frames_with_faces 
            if f.classification
        ]
    
    @property
    def average_fake_score(self) -> float:
        """Calculate average fake probability across all analyzed faces."""
        scores = self.fake_scores
        return sum(scores) / len(scores) if scores else 0.0
    
    @property
    def max_fake_score(self) -> float:
        """Get maximum fake probability detected."""
        scores = self.fake_scores
        return max(scores) if scores else 0.0
    
    @property
    def min_fake_score(self) -> float:
        """Get minimum fake probability detected."""
        scores = self.fake_scores
        return min(scores) if scores else 0.0


class DetectionPipeline:
    """
    End-to-end detection pipeline for deepfake video analysis.
    
    Orchestrates:
    1. Video loading and frame extraction
    2. Face detection on each frame
    3. Deepfake classification on detected faces
    
    Produces a VideoAnalysis object with complete results.
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        sample_rate: float = 1.0,
        use_cuda: bool = False,
        max_frames: Optional[int] = None
    ):
        """
        Initialize the detection pipeline.
        
        Args:
            weights_path: Path to classifier weights
            sample_rate: Frame sampling rate (fps)
            use_cuda: Whether to use GPU
            max_frames: Maximum frames to analyze (optional)
        """
        self.max_frames = max_frames
        
        # Initialize components
        logger.info("Initializing detection pipeline...")
        self.video_processor = VideoProcessor(sample_rate=sample_rate)
        self.face_detector = FaceDetector()
        self.classifier = DeepfakeClassifier(
            weights_path=weights_path,
            use_cuda=use_cuda
        )
        logger.info("Detection pipeline ready!")
    
    def analyze_video(
        self, 
        video_path: str,
        show_progress: bool = True
    ) -> VideoAnalysis:
        """
        Analyze a video for deepfakes.
        
        Args:
            video_path: Path to the video file
            show_progress: Whether to show progress bar
            
        Returns:
            VideoAnalysis object with complete results
        """
        # Validate video
        is_valid, error_msg = self.video_processor.validate_video(video_path)
        if not is_valid:
            raise ValueError(f"Invalid video: {error_msg}")
        
        # Get metadata
        metadata = self.video_processor.get_metadata(video_path)
        logger.info(f"Analyzing: {video_path}")
        logger.info(f"Duration: {metadata.duration_seconds:.1f}s, FPS: {metadata.fps}")
        
        # Extract and process frames
        frame_analyses = []
        frames = self.video_processor.extract_frames(video_path, self.max_frames)
        
        # Create progress wrapper
        if show_progress:
            frames = tqdm(frames, desc="Analyzing frames", unit="frame")
        
        for frame_idx, frame in frames:
            analysis = self._analyze_frame(frame, frame_idx)
            frame_analyses.append(analysis)
        
        return VideoAnalysis(
            video_path=video_path,
            metadata=metadata,
            frame_analyses=frame_analyses
        )
    
    def _analyze_frame(self, frame, frame_idx: int) -> FrameAnalysis:
        """
        Analyze a single frame.
        
        Args:
            frame: OpenCV frame image
            frame_idx: Frame index in video
            
        Returns:
            FrameAnalysis object
        """
        # Detect face
        face = self.face_detector.detect_largest_face(frame)
        
        if face is None:
            return FrameAnalysis(
                frame_index=frame_idx,
                face_detected=False
            )
        
        # Classify face
        classification = self.classifier.classify(face.cropped_face)
        
        return FrameAnalysis(
            frame_index=frame_idx,
            face_detected=True,
            face_bbox=face.bbox,
            classification=classification
        )
    
    def quick_check(self, video_path: str, num_frames: int = 5) -> ClassificationResult:
        """
        Quick deepfake check using limited frames.
        
        Args:
            video_path: Path to video
            num_frames: Number of frames to check
            
        Returns:
            Classification result based on average of checked frames
        """
        old_max = self.max_frames
        self.max_frames = num_frames
        
        analysis = self.analyze_video(video_path, show_progress=False)
        
        self.max_frames = old_max
        
        avg_fake = analysis.average_fake_score
        avg_real = 1.0 - avg_fake
        prediction = "FAKE" if avg_fake > 0.5 else "REAL"
        
        return ClassificationResult(
            prediction=prediction,
            real_probability=avg_real,
            fake_probability=avg_fake,
            confidence=max(avg_real, avg_fake)
        )
