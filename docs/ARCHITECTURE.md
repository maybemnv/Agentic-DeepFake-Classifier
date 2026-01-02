# 🧠 Agentic Deepfake Classifier - How It Works

A beginner-friendly guide to understanding what we built and how all the pieces fit together.

---

## 📖 The Big Picture

Think of this system like a **security guard** checking if a video is real or fake:

```
Video File
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUR SYSTEM                               │
│                                                             │
│  1. VIDEO PROCESSOR    → Breaks video into frames           │
│         ↓                                                   │
│  2. FACE DETECTOR      → Finds faces in each frame          │
│         ↓                                                   │
│  3. CLASSIFIER         → Checks if each face is fake        │
│         ↓                                                   │
│  4. DECISION AGENT     → Makes final verdict (Real/Fake)    │
│         ↓                                                   │
│  5. COGNITIVE AGENT    → Explains the decision in English   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    ↓
Result: "This video is FAKE with 85% confidence"
```

---

## 🗂️ Project Structure Explained

```
Agentic DeepFake Classifier/
│
├── src/                          # 🧠 The brain of our system
│   ├── analyzer.py               # Main entry point - ties everything together
│   │
│   ├── detector/                 # 🔍 Detection modules (finding & analyzing)
│   │   ├── video_processor.py    # Handles video files
│   │   ├── face_detector.py      # Finds faces using AI
│   │   ├── classifier.py         # Determines real vs fake
│   │   └── pipeline.py           # Connects all detector modules
│   │
│   └── agents/                   # 🤖 Agentic modules (decision making)
│       ├── decision_agent.py     # Makes the final call
│       └── cognitive_agent.py    # Explains in human language
│
├── model/                        # 🧪 Pre-trained AI model
│   └── ffpp_c23.pth              # The "brain" trained on 1M+ fake videos
│
├── yoink/Deepfake-Detection/     # 📦 External code we're using
│   └── network/xception.py       # The neural network architecture
│
├── frontend/                     # 🖥️ Web interface
│   └── app.py                    # Streamlit UI
│
└── main.py                       # 🚀 Command-line interface
```

---

## 🔍 Each Module Explained

### 1. Video Processor (`video_processor.py`)

**What it does:** Opens a video file and extracts individual frames (images).

**Why we need it:** AI models can't process videos directly - they need individual images.

```python
# Simple example of what it does:
video = "my_video.mp4"  # 30 second video at 30 fps = 900 frames

# We don't need ALL frames, so we sample 1 per second = 30 frames
# This is much faster and still accurate!
frames = video_processor.extract_frames(video)  # Returns 30 images
```

**Key features:**

- Validates video format (MP4, AVI, etc.)
- Configurable sampling rate (default: 1 frame per second)
- Memory efficient (uses generators)

---

### 2. Face Detector (`face_detector.py`)

**What it does:** Finds human faces in each frame.

**Why we need it:** Deepfakes manipulate FACES, so we need to isolate them.

```python
# For each frame, find the face:
frame = load_image("frame_001.jpg")

face = face_detector.detect_largest_face(frame)
# Returns: cropped 299x299 image of just the face
```

**Key features:**

- Uses dlib library (industry standard for face detection)
- Scales bounding box 1.3x to capture more context around face
- Handles frames with no faces (skips them)

---

### 3. Classifier (`classifier.py`)

**What it does:** Looks at a face image and predicts if it's REAL or FAKE.

**Why we need it:** This is the actual "detection" part!

```python
# The classifier is a neural network called XceptionNet:
face_image = load_face("face_001.jpg")

result = classifier.classify(face_image)
# Returns:
#   - real_probability: 0.15 (15% chance it's real)
#   - fake_probability: 0.85 (85% chance it's fake)
```

**How it works (simplified):**

1. The XceptionNet was trained on 1,000,000+ images of real and fake faces
2. It learned to spot tiny patterns that distinguish fakes:
   - Unnatural skin textures
   - Weird lighting reflections in eyes
   - Blending artifacts around face edges
3. When you give it a new face, it compares to what it learned

**The weights file (`ffpp_c23.pth`):**

- This is the "memory" of the trained model
- Contains millions of numbers that encode what the model learned
- Without this file, the model would be useless (random guessing)

---

### 4. Pipeline (`pipeline.py`)

**What it does:** Connects Video → Face → Classifier into one smooth flow.

```python
# Instead of calling each module separately:
pipeline = DetectionPipeline()
analysis = pipeline.analyze_video("suspicious_video.mp4")

# Returns analysis with ALL frame results:
# - Frame 1: face detected, 82% fake
# - Frame 2: face detected, 79% fake
# - Frame 3: no face detected, skipped
# - Frame 4: face detected, 85% fake
# ... etc
```

---

### 5. Decision Agent (`decision_agent.py`)

**What it does:** Takes all the frame scores and makes ONE final decision.

**Why "agentic"?** It acts autonomously - you don't tell it what to decide, it figures it out based on rules.

```python
# The agent receives scores from all frames:
frame_scores = [0.82, 0.79, 0.85, 0.88, 0.81]  # All high = likely fake

decision = decision_agent.decide(frame_scores)
# Returns:
#   - verdict: "FAKE"
#   - confidence: 0.91 (91% sure)
```

**Decision rules (from your POC):**

| Average Fake Score | Verdict       |
| ------------------ | ------------- |
| >= 0.7 (70%)       | 🚨 FAKE       |
| 0.4 to 0.7         | ⚠️ SUSPICIOUS |
| < 0.4 (40%)        | ✅ REAL       |

**Also considers:**

- **Variance:** If scores jump around a lot (0.2, 0.9, 0.3), it's less confident
- **Sample size:** More faces analyzed = more reliable decision

---

### 6. Cognitive Agent (`cognitive_agent.py`)

**What it does:** Translates technical results into human-readable explanations.

```python
# Takes the decision:
decision = DecisionResult(verdict="FAKE", confidence=0.91, ...)

response = cognitive_agent.generate_response(decision)
# Returns:
#   verdict_text: "This video shows strong indicators of deepfake manipulation."
#   recommendation: "Do not trust this video's authenticity. Verify with original source."
```

**Why we need it:** Numbers are hard to interpret. Humans need context and advice.

---

### 7. Analyzer (`analyzer.py`)

**What it does:** The "main brain" that orchestrates everything.

```python
# This is what you actually use:
from src.analyzer import DeepfakeAnalyzer

analyzer = DeepfakeAnalyzer()
result = analyzer.analyze("video.mp4")

print(result)  # Prints everything nicely formatted
```

**It combines:**

1. Detection Pipeline (video → frames → faces → scores)
2. Decision Agent (scores → verdict)
3. Cognitive Agent (verdict → explanation)

---

## 🔄 Complete Flow Example

Let's trace through what happens when you analyze a video:

```
INPUT: suspicious_video.mp4 (10 seconds, 30fps)

STEP 1: Video Processor
├── Opens video with OpenCV
├── Extracts 10 frames (1 per second)
└── Returns: [frame_0, frame_1, ..., frame_9]

STEP 2: Face Detector (for each frame)
├── Frame 0: Found face at (120, 80), cropped to 299x299
├── Frame 1: Found face at (125, 82), cropped to 299x299
├── Frame 2: No face detected (person looked away)
├── ... (continues for all frames)
└── Returns: [face_0, face_1, face_3, ...]  (8 faces total)

STEP 3: Classifier (for each face)
├── Face 0: real=0.18, fake=0.82 → "FAKE"
├── Face 1: real=0.21, fake=0.79 → "FAKE"
├── Face 3: real=0.15, fake=0.85 → "FAKE"
├── ...
└── Returns: [0.82, 0.79, 0.85, 0.88, 0.81, 0.77, 0.83, 0.86]

STEP 4: Decision Agent
├── Calculates average: 0.826 (82.6%)
├── Checks threshold: 0.826 >= 0.7 → FAKE
├── Calculates confidence: 0.89 (89%)
└── Returns: verdict=FAKE, confidence=89%

STEP 5: Cognitive Agent
├── Looks up template for "FAKE" + "high confidence"
├── Generates explanation with statistics
└── Returns: "This video shows strong indicators of deepfake manipulation..."

OUTPUT:
============================================================
DEEPFAKE ANALYSIS RESULT
============================================================

📁 Video: suspicious_video.mp4
⏱️  Duration: 10.0s

🚨 VERDICT: FAKE
📊 Confidence: 89%

--- Explanation ---
This video shows strong indicators of deepfake manipulation.

--- Technical Summary ---
• Frames analyzed: 10
• Faces detected: 8
• Avg fake score: 82.6%
• Score range: 77% - 88%

--- Recommendation ---
Do not trust this video's authenticity.
Verify with the original source if possible.
============================================================
```

---

## 🧪 The AI Model Explained

### What is XceptionNet?

XceptionNet is a type of **Convolutional Neural Network (CNN)** - a special AI architecture designed for image analysis.

**Think of it like this:**

- A human learns to spot fakes by looking at thousands of examples
- XceptionNet does the same, but with millions of examples and math

**Why XceptionNet for deepfakes?**

- It was designed to find subtle patterns in images
- FaceForensics++ researchers found it works really well for deepfakes
- It's accurate AND reasonably fast

### The Pre-trained Weights (`ffpp_c23.pth`)

```
ffpp = FaceForensics++ (the dataset it was trained on)
c23 = Compression level 23 (high quality videos)
.pth = PyTorch format
```

This 83MB file contains **~22 million numbers** that represent what the model learned:

- Patterns in real faces
- Patterns in fake faces
- The differences between them

**Without this file, the model is just an empty shell that guesses randomly.**

---

## 🤖 What Makes It "Agentic"?

Traditional software: "If score > 0.7, return FAKE"

**Agentic approach:**

1. **Autonomous reasoning:** The Decision Agent considers multiple factors (average, variance, sample size)
2. **Confidence awareness:** It knows when it's unsure
3. **Adaptive thresholds:** Can adjust based on context
4. **Self-explanation:** Generates its own reasoning (Cognitive Agent)

This is more like how a human expert would work - not just following rigid rules, but understanding context and explaining decisions.

---

## 🚀 How to Use It

### Option 1: Command Line

```bash
# Basic analysis
python main.py --video path/to/video.mp4

# Quick check (faster, less accurate)
python main.py --video path/to/video.mp4 --quick

# Save results to file
python main.py --video path/to/video.mp4 --output results.json
```

### Option 2: Streamlit Web UI

```bash
streamlit run frontend/app.py
# Then open http://localhost:8501 in your browser
```

### Option 3: Python Code

```python
from src.analyzer import DeepfakeAnalyzer

# Create analyzer (loads model once)
analyzer = DeepfakeAnalyzer()

# Analyze any video
result = analyzer.analyze("video.mp4")

# Access results
print(result.verdict)       # REAL, FAKE, SUSPICIOUS, or INCONCLUSIVE
print(result.confidence)    # 0.0 to 1.0
print(result.explanation)   # Human-readable text
```

---

## ❓ Common Questions

### Q: How accurate is it?

The XceptionNet model achieves ~95% accuracy on the FaceForensics++ test set. Real-world accuracy may vary depending on:

- Video quality
- Type of deepfake
- Compression level

### Q: Can it be fooled?

Yes, adversarial attacks can sometimes fool AI detectors. This is why we:

- Analyze multiple frames
- Report confidence levels
- Recommend manual verification for important cases

### Q: Why is it slow?

The neural network processes each face through 36 convolutional layers with millions of calculations. To speed up:

- Use `--quick` flag (analyzes fewer frames)
- Use `--max-frames 10` to limit analysis
- Use GPU (`--cuda`) if available

### Q: What video formats work?

MP4, AVI, MOV, MKV, WebM

---

## 📚 Key Terms Glossary

| Term             | Definition                                                     |
| ---------------- | -------------------------------------------------------------- |
| **Deepfake**     | AI-generated fake video, usually swapping faces                |
| **XceptionNet**  | Neural network architecture optimized for image classification |
| **dlib**         | Library for face detection and recognition                     |
| **CNN**          | Convolutional Neural Network - AI for processing images        |
| **Inference**    | Running a trained model to make predictions                    |
| **Weights**      | The learned parameters of a neural network                     |
| **Frame**        | A single image from a video                                    |
| **Bounding Box** | Rectangle coordinates around a detected face                   |
| **Softmax**      | Function that converts model output to probabilities           |

---

## 🏗️ Architecture Diagram

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

_Created for the Agentic Deepfake Classifier POC_
