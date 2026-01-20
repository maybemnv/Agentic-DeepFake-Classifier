from fastapi import Request
def get_classifier(request: Request) -> DeepfakeClassifier:
    return request.app.state.classifier