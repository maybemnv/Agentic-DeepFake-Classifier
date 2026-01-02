# 🔍 Agentic Deepfake Classifier

An autonomous AI-powered deepfake detection system that analyzes video authenticity using XceptionNet and agentic decision-making.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff69b4.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Features

- **Video Analysis**: Upload any video and get instant deepfake detection results
- **Agentic Decision Making**: Autonomous verdict determination with confidence scoring
- **Human-Readable Explanations**: Clear explanations of why a video is classified as real or fake
- **Multiple Interfaces**: CLI, Python API, and beautiful Streamlit web UI
- **Offline Capable**: Runs entirely on your local machine, no cloud dependency
- **CPU Optimized**: Works on regular laptops without GPU

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Agentic-DeepFake-Classifier.git
cd Agentic-DeepFake-Classifier

# Install dependencies (using uv)
uv add torch torchvision opencv-python pillow numpy tqdm streamlit
uv add dlib-bin  # Pre-built dlib for Windows

# Or using pip
pip install -r requirements.txt
```

### Download Pre-trained Weights

Download the FaceForensics++ pretrained weights from:

- [Google Drive - FF++ Weights](https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8)

Save as `model/ffpp_c23.pth`

### Run the Web UI

```bash
streamlit run frontend/app.py
```

Then open http://localhost:8501 in your browser.

### Run via CLI

```bash
# Full analysis
python main.py --video path/to/video.mp4

# Quick check (faster)
python main.py --video path/to/video.mp4 --quick

# Save results to JSON
python main.py --video path/to/video.mp4 --output results.json
```

---

## How It Works

```
Video Input
    ↓
┌─────────────────────────────────────────┐
│  1. Frame Extraction (OpenCV)           │
│  2. Face Detection (dlib)               │
│  3. Deepfake Classification (XceptionNet)│
│  4. Decision Agent (Agentic Reasoning)   │
│  5. Cognitive Agent (Explanations)       │
└─────────────────────────────────────────┘
    ↓
Verdict: REAL / FAKE / SUSPICIOUS
+ Confidence Score + Explanation
```

### Decision Thresholds

| Fake Score | Verdict       | Meaning                                  |
| ---------- | ------------- | ---------------------------------------- |
| >= 70%     | 🚨 FAKE       | High likelihood of deepfake manipulation |
| 40-70%     | ⚠️ SUSPICIOUS | Warrants further investigation           |
| < 40%      | ✅ REAL       | Appears authentic                        |

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                            │
├────────────────────────────────────────────────────────────────────────┤
│   main.py (CLI)          │        frontend/app.py (Streamlit)         │
│   - Command line args    │        - File upload                       │
│   - JSON output          │        - Visual results                    │
└──────────────┬───────────┴────────────────┬────────────────────────────┘
               │                            │
               └──────────┬─────────────────┘
                          ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         ANALYZER (src/analyzer.py)                      │
│                    Orchestrates the entire flow                         │
└────────────────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌────────────┐ ┌─────────────────┐
│ Detection       │ │ Decision   │ │ Cognitive       │
│ Pipeline        │ │ Agent      │ │ Agent           │
│ ┌─────────────┐ │ │            │ │                 │
│ │Video Proc.  │ │ │ Scores →   │ │ Decision →      │
│ └─────────────┘ │ │ Verdict    │ │ Explanation     │
│ ┌─────────────┐ │ │            │ │                 │
│ │Face Detect. │ │ │ Thresholds │ │ Templates       │
│ └─────────────┘ │ │ Confidence │ │ Recommendations │
│ ┌─────────────┐ │ │            │ │                 │
│ │Classifier   │ │ │            │ │                 │
│ └─────────────┘ │ │            │ │                 │
└─────────────────┘ └────────────┘ └─────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    MODEL LAYER (yoink/Deepfake-Detection/)              │
│                                                                         │
│   network/xception.py  ←→  model/ffpp_c23.pth (weights)                │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Acknowledgments & Credits

This project builds upon the excellent work of the FaceForensics++ team and the deepfake detection community.

### Model Architecture & Weights

**FaceForensics++: Learning to Detect Manipulated Facial Images**

```bibtex
@inproceedings{roessler2019faceforensicspp,
    author = {Andreas Rössler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nießner},
    title = {FaceForensics++: Learning to Detect Manipulated Facial Images},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year = {2019}
}
```

- **Paper**: [FaceForensics++](https://github.com/ondyari/FaceForensics)
- **Authors**: Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner
- **Institution**: Technical University of Munich

### PyTorch Implementation

**Deepfake-Detection (PyTorch)**

- **Repository**: [HongguLiu/Deepfake-Detection](https://github.com/HongguLiu/Deepfake-Detection)
- **Author**: Honggu Liu
- **Description**: PyTorch implementation of deepfake detection using XceptionNet, based on FaceForensics++

### XceptionNet Architecture

```bibtex
@inproceedings{chollet2017xception,
    author = {François Chollet},
    title = {Xception: Deep Learning with Depthwise Separable Convolutions},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2017}
}
```

### Face Detection

- **dlib**: Davis E. King - [dlib.net](http://dlib.net/)

---

## License

This project is for educational and research purposes only.

- The code in this repository is MIT licensed
- Pre-trained model weights are subject to [FaceForensics++ terms](https://github.com/ondyari/FaceForensics)
- For commercial use, please contact the original authors
