"""
Face Detection Module
Handles face detection and cropping using dlib.
"""

import cv2
import dlib
import numpy as np
from typing import List, Optional
import logging

from ..core import FaceResult, FaceDetectionConfig, FACE_CONFIG

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects faces in images using dlib frontal face detector.
    """
    
    def __init__(self, config: FaceDetectionConfig = None):
        """
        Initialize the face detector.
        
        Args:
            config: Face detection configuration
        """
        self.config = config or FACE_CONFIG
        self.detector = dlib.get_frontal_face_detector()
        logger.info("Initialized dlib frontal face detector")
    
    def _get_scaled_bbox(
        self, 
        face: dlib.rectangle, 
        width: int, 
        height: int
    ) -> tuple:
        """Get scaled bounding box from dlib face detection."""
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        
        face_size = max(x2 - x1, y2 - y1)
        size = int(face_size * self.config.scale_factor)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        new_x1 = max(0, center_x - size // 2)
        new_y1 = max(0, center_y - size // 2)
        
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
            max_faces: Maximum number of faces to return
            
        Returns:
            List of FaceResult objects
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        faces = self.detector(gray, 1)
        
        results = []
        for face in faces:
            face_width = face.right() - face.left()
            face_height = face.bottom() - face.top()
            
            if face_width < self.config.min_face_size or face_height < self.config.min_face_size:
                continue
            
            x, y, w, h = self._get_scaled_bbox(face, width, height)
            cropped = image[y:y+h, x:x+w]
            
            if cropped.size == 0:
                continue
            
            cropped_resized = cv2.resize(cropped, self.config.target_size)
            
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
        """Detect the largest face in an image."""
        faces = self.detect_faces(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    
    def has_face(self, image: np.ndarray) -> bool:
        """Quick check if image contains any faces."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        return len(faces) > 0
