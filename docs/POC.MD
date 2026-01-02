# Agentic AI Deepfake Detection PoC

## 1. PoC Objective

Build a working, offline, edge runnable proof of concept that demonstrates autonomous detection and verification of deepfake videos. The system must analyze video input locally, make decisions without cloud dependency, and generate an authenticity verdict with reasoning suitable for field operatives.

This PoC prioritizes credibility, clarity, and completeness over scale.

---

## 2. Problem Scope (Explicitly Scoped for Hackathon)

### In Scope

- Video deepfake detection only
- Offline inference on CPU
- Single agent with autonomous decision logic
- Local explanation generation
- Demo via CLI or Streamlit

### Out of Scope

- Real time live camera feed
- Model training
- Mobile application packaging
- Multi agent orchestration
- Audio deepfake detection

---

## 3. Success Criteria

The PoC is considered successful if it can:

1. Accept a video input
2. Detect faces and analyze authenticity locally
3. Output a clear verdict: Real, Fake, or Suspicious
4. Provide a confidence score
5. Provide a human readable explanation
6. Run fully offline

---

## 4. System Architecture

### High Level Flow

Video Input
→ Frame Sampler
→ Face Detection
→ Face Cropping
→ Deepfake Classifier
→ Decision Agent
→ Cognitive Response Generator
→ Output

---

## 5. Component Breakdown

### 5.1 Video Ingestion Module

Responsibilities:

- Load video from local filesystem
- Validate format and length

Implementation:

- OpenCV VideoCapture

Output:

- Video stream object

---

### 5.2 Frame Sampling Module

Responsibilities:

- Reduce computational load
- Sample frames at fixed intervals

Implementation:

- Extract 1 frame per second
- Configurable sampling rate

Output:

- List of frames

---

### 5.3 Face Detection Module

Responsibilities:

- Detect faces in sampled frames
- Ensure robustness to pose and lighting

Implementation:

- dlib frontal face detector (from FaceForensics++ pipeline)
- Bounding box scaling for better face crops

Output:

- Bounding boxes
- Cropped face images (scaled 1.3x for context)

---

### 5.4 Deepfake Classification Module

Responsibilities:

- Classify cropped faces as real or fake

Implementation:

- XceptionNet architecture (from FaceForensics++ PyTorch implementation)
- Pre-trained weights: FF++\_c23.pth (trained on FaceForensics++ dataset)
- PyTorch inference only (CPU mode for edge deployment)
- Input: 299x299 RGB face crops

Model Source:

- Architecture: yoink/Deepfake-Detection/network/xception.py
- Weights: https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8

Output:

- Softmax probabilities [real_prob, fake_prob] per frame

---

### 5.5 Decision Agent Module (Agentic Core)

Responsibilities:

- Aggregate frame level predictions
- Make autonomous authenticity decision

Decision Logic:

- If average fake score ≥ 0.7 → Fake
- If 0.4 ≤ score < 0.7 → Suspicious
- If score < 0.4 → Real

Output:

- Final verdict
- Confidence score

---

### 5.6 Cognitive Assistance Module

Responsibilities:

- Convert numeric outputs into human readable insights

Implementation:

- Rule based explanation templates

Example Outputs:

- "High likelihood of deepfake detected due to facial inconsistencies"
- "Video appears authentic with stable facial dynamics"

---

### 5.7 Output Interface

Responsibilities:

- Display results clearly

Implementation Options:

- CLI print output
- Streamlit UI

Output Fields:

- Verdict
- Confidence
- Explanation

---

## 6. Security and Edge Constraints (Conceptual)

This PoC simulates:

- On device execution
- No network calls
- Minimal memory footprint

Future Extensions:

- Secure boot
- RBAC
- Firmware signing

---

## 7. Time Bound Execution Checklist (6 PM to 10 PM)

### 6:00 to 6:30 Setup

- Create repo structure
- Setup virtual environment
- Install dependencies

### 6:30 to 7:15 Video + Frame Pipeline

- Load video
- Implement frame sampler
- Save debug frames

### 7:15 to 8:00 Face Detection

- Integrate face detector
- Crop faces
- Verify face outputs visually

### 8:00 to 8:45 Deepfake Model Integration

- Load pretrained model
- Run inference on face crops
- Print probabilities

### 8:45 to 9:15 Decision Agent

- Aggregate predictions
- Implement verdict logic

### 9:15 to 9:40 Cognitive Output

- Add explanation mapping
- Clean output formatting

### 9:40 to 10:00 Proof Generation

- Take screenshots
- Record short demo
- Write README summary

---

## 8. Deliverables

Mandatory:

- Working codebase
- Demo output screenshots
- Architecture explanation

Optional:

- Short demo video
- Model performance note

---

## 9. Risk Mitigation

Risk: Model not detecting properly
Mitigation: Use known fake and real samples

Risk: Performance slow
Mitigation: Reduce frame sampling

Risk: Face detection failure
Mitigation: Skip frame with no face
