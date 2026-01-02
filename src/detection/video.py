"""
Video Processing Module
Handles video loading, validation, and frame extraction.
"""

import cv2
import os
from typing import Generator, Optional, Tuple, List
import logging

from ..core import VideoMetadata, VideoConfig, VIDEO_CONFIG
from ..core.exceptions import VideoNotFoundError, VideoFormatError, VideoCorruptedError

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes video files for deepfake detection.
    
    Responsibilities:
    - Load video from filesystem
    - Validate format and integrity
    - Extract frames at configurable intervals
    """
    
    def __init__(self, config: VideoConfig = None):
        """
        Initialize the video processor.
        
        Args:
            config: Video processing configuration
        """
        self.config = config or VIDEO_CONFIG
    
    def validate(self, video_path: str) -> None:
        """
        Validate that video file exists and is in supported format.
        
        Args:
            video_path: Path to the video file
            
        Raises:
            VideoNotFoundError: If file doesn't exist
            VideoFormatError: If format not supported
            VideoCorruptedError: If file is corrupted
        """
        if not os.path.exists(video_path):
            raise VideoNotFoundError(f"Video file not found: {video_path}")
        
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.config.supported_formats:
            raise VideoFormatError(
                f"Unsupported format: {ext}. Supported: {self.config.supported_formats}"
            )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoCorruptedError("Failed to open video file")
        
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            raise VideoCorruptedError("Video file appears to be empty or corrupted")
    
    def get_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata object
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return VideoMetadata(
            path=video_path,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            width=width,
            height=height,
            format=os.path.splitext(video_path)[1].lower()
        )
    
    def extract_frames(
        self, 
        video_path: str, 
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, 'cv2.Mat'], None, None]:
        """
        Extract frames from video at the configured sample rate.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Yields:
            Tuple of (frame_index, frame_image)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_interval = int(fps / self.config.sample_rate) if fps > 0 else 1
        frame_interval = max(1, frame_interval)
        
        max_to_extract = max_frames or self.config.max_frames
        
        frame_count = 0
        extracted_count = 0
        
        logger.info(f"Extracting frames at {self.config.sample_rate} fps")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                yield frame_count, frame
                extracted_count += 1
                
                if max_to_extract and extracted_count >= max_to_extract:
                    break
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {extracted_count} frames")
    
    def extract_frames_list(
        self, 
        video_path: str, 
        max_frames: Optional[int] = None
    ) -> List[Tuple[int, 'cv2.Mat']]:
        """Extract frames as a list."""
        return list(self.extract_frames(video_path, max_frames))
