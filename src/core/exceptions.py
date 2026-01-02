"""
Custom Exceptions Module
All project-specific exceptions.
"""


class DeepfakeDetectorError(Exception):
    """Base exception for all project errors."""
    pass


class VideoError(DeepfakeDetectorError):
    """Video-related errors."""
    pass


class VideoNotFoundError(VideoError):
    """Video file not found."""
    pass


class VideoFormatError(VideoError):
    """Unsupported video format."""
    pass


class VideoCorruptedError(VideoError):
    """Video file is corrupted."""
    pass


class FaceDetectionError(DeepfakeDetectorError):
    """Face detection errors."""
    pass


class NoFacesDetectedError(FaceDetectionError):
    """No faces found in video."""
    pass


class ClassifierError(DeepfakeDetectorError):
    """Classifier errors."""
    pass


class ModelNotFoundError(ClassifierError):
    """Model weights file not found."""
    pass


class ModelLoadError(ClassifierError):
    """Failed to load model."""
    pass
