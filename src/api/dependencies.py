from typing import TYPE_CHECKING
from fastapi import Request

if TYPE_CHECKING:
    from ..detection import DeepfakeClassifier

def get_classifier(request: Request) -> "DeepfakeClassifier":
    """Dependency to get the shared classifier instance from app state."""
    return request.app.state.classifier