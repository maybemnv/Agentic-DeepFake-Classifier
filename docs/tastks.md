# Agentic Deepfake Detection - Implementation Tasks

## Tech Stack Decision ✅

**Model**: XceptionNet (PyTorch) from FaceForensics++ implementation
**Weights**: ffpp_c23.pth (pre-trained on FaceForensics++ dataset)
**Face Detection**: dlib frontal face detector
**Framework**: PyTorch (inference only, no training)

---

## Phase 1: Environment & Model Setup ✅ COMPLETED

### Task 1.1: Project Structure (10 mins)

- [x] Verify project directory structure
- [x] Ensure yoink/Deepfake-Detection/ is intact (needed for model architecture)

### Task 1.2: Download Pre-trained Weights (5 mins)

- [x] Download weights from Google Drive
- [x] Saved to: `model/ffpp_c23.pth` (83MB)

### Task 1.3: Install Dependencies (10 mins)

- [ ] Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Or install individually:
  ```bash
  pip install torch torchvision opencv-python dlib pillow tqdm numpy streamlit
  ```

### Task 1.4: Test Model Loading (10 mins)

- [ ] Run: `python -c "from src.detector import DeepfakeClassifier; c = DeepfakeClassifier()"`
- [ ] Verify no errors

**Checkpoint 1**: Model loads correctly ✓

---

## Phase 2: Detection Pipeline ✅ COMPLETED

### Task 2.1: Video Ingestion Module

- [x] `src/detector/video_processor.py` - Video loading, validation, frame extraction

### Task 2.2: Face Detection Module

- [x] `src/detector/face_detector.py` - dlib face detection with scaled bounding boxes

### Task 2.3: Classification Module

- [x] `src/detector/classifier.py` - XceptionNet deepfake classification

### Task 2.4: Pipeline Integration

- [x] `src/detector/pipeline.py` - End-to-end detection pipeline

**Checkpoint 2**: Pipeline outputs predictions per frame ✓

---

## Phase 3: Agentic Decision Layer ✅ COMPLETED

### Task 3.1: Decision Agent

- [x] `src/agents/decision_agent.py` - Autonomous verdict determination

### Task 3.2: Cognitive Response Generator

- [x] `src/agents/cognitive_agent.py` - Human-readable explanations

### Task 3.3: Main Analysis Interface

- [x] `src/analyzer.py` - Complete analyzer combining all components

**Checkpoint 3**: Agentic analysis works end-to-end ✓

---

## Phase 4: User Interface ✅ COMPLETED

### Task 4.1: CLI Interface

- [x] `main.py` - Full CLI with argparse, JSON output, quick mode

### Task 4.2: Streamlit UI

- [x] `frontend/app.py` - Beautiful web UI with:
  - Video upload
  - Progress indicators
  - Verdict display with color coding
  - Detailed explanations
  - JSON report download

**Checkpoint 4**: Working UI ✓

---

## Phase 5: Testing & Documentation

### Task 5.1: Test with Sample Videos (15 mins)

- [ ] Test on known fake videos
- [ ] Test on known real videos
- [ ] Verify accuracy is reasonable

### Task 5.2: Documentation (15 mins)

- [ ] Update README.md
- [ ] Add architecture diagram

### Task 5.3: Demo Recording (10 mins)

- [ ] Record screen demo
- [ ] Take screenshots

**Checkpoint 5**: POC ready for submission ✓

---

## Project Structure

```
Agentic DeepFake Classifier/
├── src/
│   ├── __init__.py
│   ├── analyzer.py              # Main analyzer (entry point)
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── video_processor.py   # Video loading & frame extraction
│   │   ├── face_detector.py     # dlib face detection
│   │   ├── classifier.py        # XceptionNet classification
│   │   └── pipeline.py          # End-to-end pipeline
│   └── agents/
│       ├── __init__.py
│       ├── decision_agent.py    # Verdict determination
│       └── cognitive_agent.py   # Human explanations
├── model/
│   └── ffpp_c23.pth             # Pre-trained weights (83MB)
├── yoink/Deepfake-Detection/    # FaceForensics++ model architecture
├── frontend/
│   └── app.py                   # Streamlit web UI
├── main.py                      # CLI interface
├── requirements.txt
└── docs/
    ├── POC.MD
    └── tastks.md
```

---

## Quick Start

### CLI Usage:

```bash
# Full analysis
python main.py --video path/to/video.mp4

# Quick check (5 frames only)
python main.py --video path/to/video.mp4 --quick

# Save results to JSON
python main.py --video path/to/video.mp4 --output results.json

# Verbose output
python main.py --video path/to/video.mp4 --verbose
```

### Streamlit UI:

```bash
streamlit run frontend/app.py
```

### Python API:

```python
from src.analyzer import DeepfakeAnalyzer

analyzer = DeepfakeAnalyzer()
result = analyzer.analyze("video.mp4")

print(result)              # Pretty printed output
print(result.verdict)      # REAL, FAKE, SUSPICIOUS, or INCONCLUSIVE
print(result.confidence)   # 0.0 to 1.0
print(result.explanation)  # Human-readable explanation
```

---

## Decision Thresholds

| Score Range | Verdict         | Color  |
| ----------- | --------------- | ------ |
| >= 0.7      | 🚨 FAKE         | Red    |
| 0.4 - 0.7   | ⚠️ SUSPICIOUS   | Yellow |
| < 0.4       | ✅ REAL         | Green  |
| No faces    | ❓ INCONCLUSIVE | Gray   |
