import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_model_load")

try:
    logger.info("Importing classifier module...")
    from src.detection.classifier import DeepfakeClassifier

    logger.info("Initializing classifier (this should trigger model download/load)...")
    classifier = DeepfakeClassifier(use_cuda=False)

    logger.info("Model loaded successfully!")

    import numpy as np

    dummy_face = np.zeros((300, 300, 3), dtype=np.uint8)

    logger.info("Testing inference on dummy input...")
    result = classifier.classify(dummy_face)
    logger.info(f"Inference Result: {result}")

except Exception as e:
    logger.error(f"Test failed: {e}")
    import traceback

    traceback.print_exc()
