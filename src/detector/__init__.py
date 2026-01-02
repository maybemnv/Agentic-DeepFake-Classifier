# Detection Pipeline Module
from .video_processor import VideoProcessor, VideoMetadata
from .face_detector import FaceDetector, FaceResult
from .classifier import DeepfakeClassifier, ClassificationResult
from .pipeline import DetectionPipeline, VideoAnalysis, FrameAnalysis

__all__ = [
    'VideoProcessor', 
    'VideoMetadata',
    'FaceDetector', 
    'FaceResult',
    'DeepfakeClassifier', 
    'ClassificationResult',
    'DetectionPipeline',
    'VideoAnalysis',
    'FrameAnalysis'
]
