# Agentic Deepfake Classifier

An autonomous deepfake detection system utilizing XceptionNet and agentic reasoning to analyze video authenticity.

## System Overview

This system implements a modular pipeline for deepfake detection, focusing on explainability and temporal consistency:

1.  **Video Processing**: Efficient frame extraction at configurable sample rates using OpenCV.
2.  **Face Detection**: Face identification and cropping using dlib.
3.  **Inference Engine**: PyTorch implementation of XceptionNet trained on FaceForensics++ (c23 compression).
4.  **Agentic Analysis**:
    - **Decision Agent**: Aggregates per-frame probabilities and applies temporal logic to determine a final verdict.
    - **Cognitive Agent**: Synthesizes technical metrics into human-readable explanations.

## Technology Stack

- **Runtime**: Python 3.10+
- **Inference**: PyTorch (CUDA supported)
- **API**: FastAPI
- **Interface**: Streamlit
- **Vision**: OpenCV, dlib

## Installation

This project utilizes `uv` for fast dependency management.

```bash
# Clone repository
git clone https://github.com/yourusername/Agentic-DeepFake-Classifier.git
cd Agentic-DeepFake-Classifier

# Install dependencies
uv sync
```

Alternatively, standard pip installation is supported:

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Launch the interactive dashboard for video analysis.

```bash
streamlit run frontend/app.py
```

Access the UI at `http://localhost:8501`.

### REST API

Start the backend server for programmatic integration.

```bash
uvicorn src.api.app:app
```

API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

### Command Line Interface

Analyze videos via the terminal.

```bash
# Output summary to console
python main.py --video data/sample.mp4

# Save detailed JSON report
python main.py --video data/sample.mp4 --output report.json

# Quick analysis (first 5 frames)
python main.py --video data/sample.mp4 --quick
```

## Project Structure

The codebase follows a modular architecture:

- `src/api`: FastAPI application, dependency injection, and routes.
- `src/core`: Domain models, configuration classes, and custom exceptions.
- `src/detection`: Deep learning models and computer vision pipelines (XceptionNet, TransferModel).
- `src/pipeline`: Orchestration layer connecting video processing, detection, and analysis.
- `src/agents`: Logic for decision making and explanation generation.
- `model/`: Directory for pre-trained model weights (`ffpp_c23.pth`).

## License and Acknowledgments

The source code is released under the MIT License.

This work builds upon the **FaceForensics++** dataset and benchmark. Usage of the pre-trained weights is subject to the FaceForensics++ license terms (non-commercial research use).

- **FaceForensics++**: Rössler et al. (ICCV 2019)
- **Xception**: Chollet (CVPR 2017)
