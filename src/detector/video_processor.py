"""
Video Processor Module
Handles video ingestion, validation, and frame extraction.

Author: Agentic Deepfake Classifier
"""

import cv2
import os
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata about the processed video."""
    path: str
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
    format: str


class VideoProcessor:
    """
    Processes video files for deepfake detection.
    
    Responsibilities:
    - Load video from filesystem
    - Validate format and integrity
    - Extract frames at configurable intervals
    """
    
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def __init__(self, sample_rate: float = 1.0):
        """
        Initialize the video processor.
        
        Args:
            sample_rate: Frames to extract per second (default: 1 fps)
        """
        self.sample_rate = sample_rate
    
    def validate_video(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate that video file exists and is in supported format.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(video_path):
            return False, f"Video file not found: {video_path}"
        
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {ext}. Supported: {self.SUPPORTED_FORMATS}"
        
        # Try to open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Failed to open video file"
        
        # Check if video has frames
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            return False, "Video file appears to be empty or corrupted"
        
        return True, "Valid"
    
    def get_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata object with video information
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
            max_frames: Maximum number of frames to extract (optional)
            
        Yields:
            Tuple of (frame_index, frame_image)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame interval based on sample rate
        frame_interval = int(fps / self.sample_rate) if fps > 0 else 1
        frame_interval = max(1, frame_interval)  # At least 1
        
        frame_count = 0
        extracted_count = 0
        
        logger.info(f"Extracting frames at {self.sample_rate} fps (interval: {frame_interval})")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                yield frame_count, frame
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {extracted_count} frames from {frame_count} total")
    
    def extract_frames_list(
        self, 
        video_path: str, 
        max_frames: Optional[int] = None
    ) -> List[Tuple[int, 'cv2.Mat']]:
        """
        Extract frames as a list (non-generator version).
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract (optional)
            
        Returns:
            List of (frame_index, frame_image) tuples
        """
        return list(self.extract_frames(video_path, max_frames))
