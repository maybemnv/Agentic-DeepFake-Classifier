"""
Detection Pipeline Module
Orchestrates video processing, face detection, and classification.
"""

from typing import Optional
from tqdm import tqdm
import logging

from ..core import (
    VideoAnalysis, 
    FrameAnalysis, 
    VideoConfig, 
    VIDEO_CONFIG,
    DEFAULT_WEIGHTS_PATH
)
from ..detection import VideoProcessor, FaceDetector, DeepfakeClassifier

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    End-to-end detection pipeline for deepfake video analysis.
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
            max_frames: Maximum frames to analyze
        """
        self.max_frames = max_frames
        
        # Create config with custom sample rate
        video_config = VideoConfig(sample_rate=sample_rate, max_frames=max_frames)
        
        logger.info("Initializing detection pipeline...")
        self.video_processor = VideoProcessor(config=video_config)
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
            VideoAnalysis with complete results
        """
        # Validate video
        self.video_processor.validate(video_path)
        
        # Get metadata
        metadata = self.video_processor.get_metadata(video_path)
        logger.info(f"Analyzing: {video_path}")
        logger.info(f"Duration: {metadata.duration_seconds:.1f}s")
        
        # Extract and process frames
        frame_analyses = []
        frames = self.video_processor.extract_frames(video_path, self.max_frames)
        
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
        """Analyze a single frame."""
        face = self.face_detector.detect_largest_face(frame)
        
        if face is None:
            return FrameAnalysis(frame_index=frame_idx, face_detected=False)
        
        classification = self.classifier.classify(face.cropped_face)
        
        return FrameAnalysis(
            frame_index=frame_idx,
            face_detected=True,
            face_bbox=face.bbox,
            classification=classification
        )
