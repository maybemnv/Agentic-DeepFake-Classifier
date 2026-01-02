"""
Face Detection Module
Handles face detection and cropping using dlib.

Author: Agentic Deepfake Classifier
"""

import cv2
import dlib
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    """Result of face detection for a single face."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    cropped_face: np.ndarray
    confidence: float = 1.0  # dlib doesn't provide confidence, default to 1.0


class FaceDetector:
    """
    Detects faces in images using dlib frontal face detector.
    
    Responsibilities:
    - Detect faces in frames
    - Crop and scale faces for classification
    - Handle edge cases (no faces, multiple faces)
    """
    
    def __init__(
        self, 
        scale_factor: float = 1.3,
        min_face_size: int = 64,
        target_size: Tuple[int, int] = (299, 299)
    ):
        """
        Initialize the face detector.
        
        Args:
            scale_factor: Bounding box scale multiplier for more context
            min_face_size: Minimum face size in pixels
            target_size: Target size for cropped faces (width, height)
        """
        self.scale_factor = scale_factor
        self.min_face_size = min_face_size
        self.target_size = target_size
        
        # Initialize dlib face detector
        self.detector = dlib.get_frontal_face_detector()
        logger.info("Initialized dlib frontal face detector")
    
    def _get_scaled_bbox(
        self, 
        face: dlib.rectangle, 
        width: int, 
        height: int
    ) -> Tuple[int, int, int, int]:
        """
        Get scaled bounding box from dlib face detection.
        
        Args:
            face: dlib rectangle object
            width: Image width
            height: Image height
            
        Returns:
            Tuple of (x, y, w, h) for the scaled bounding box
        """
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        # Calculate scaled size
        face_width = x2 - x1
        face_height = y2 - y1
        size = int(max(face_width, face_height) * self.scale_factor)
        
        # Center the box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Calculate new coordinates
        new_x1 = max(0, center_x - size // 2)
        new_y1 = max(0, center_y - size // 2)
        
        # Ensure box fits within image
        size = min(size, width - new_x1, height - new_y1)
        
        return new_x1, new_y1, size, size
    
    def detect_faces(
        self, 
        image: np.ndarray, 
        max_faces: Optional[int] = None
    ) -> List[FaceResult]:
        """
        Detect all faces in an image.
        
        Args:
            image: BGR image (OpenCV format)
            max_faces: Maximum number of faces to return (optional)
            
        Returns:
            List of FaceResult objects
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        # Detect faces
        faces = self.detector(gray, 1)
        
        results = []
        for face in faces:
            # Filter by minimum size
            face_width = face.right() - face.left()
            face_height = face.bottom() - face.top()
            
            if face_width < self.min_face_size or face_height < self.min_face_size:
                continue
            
            # Get scaled bounding box
            x, y, w, h = self._get_scaled_bbox(face, width, height)
            
            # Crop face
            cropped = image[y:y+h, x:x+w]
            
            # Skip if crop is empty
            if cropped.size == 0:
                continue
            
            # Resize to target size
            cropped_resized = cv2.resize(cropped, self.target_size)
            
            results.append(FaceResult(
                bbox=(x, y, w, h),
                cropped_face=cropped_resized,
                confidence=1.0
            ))
            
            if max_faces and len(results) >= max_faces:
                break
        
        logger.debug(f"Detected {len(results)} faces")
        return results
    
    def detect_largest_face(self, image: np.ndarray) -> Optional[FaceResult]:
        """
        Detect the largest face in an image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            FaceResult for the largest face, or None if no faces detected
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Return the face with largest bounding box
        return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    
    def has_face(self, image: np.ndarray) -> bool:
        """
        Quick check if image contains any faces.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            True if at least one face is detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        return len(faces) > 0
