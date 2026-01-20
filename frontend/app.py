"""
Agentic Deepfake Classifier - Streamlit Web Interface

Modular, clean, and HTTP-only client for the Deepfake Detection API.
"""

import os
import json
import time
import requests
import tempfile
import streamlit as st
from typing import Optional, Dict, Any

# Constants
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 300

class DeepfakeClient:
    """Handles communication with the Deepfake Analysis API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def analyze(self, video_path: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uploads a video and retrieves analysis results.
        
        Args:
            video_path: Local path to the video file.
            settings: Dictionary of analysis settings (sample_rate, etc.)
        
        Returns:
            JSON response dictionary or None on failure.
        """
        url = f"{self.base_url}/analyze"
        
        try:
            with open(video_path, "rb") as f:
                files = {"file": (os.path.basename(video_path), f, "video/mp4")}
                # Convert settings to form data
                max_frames = settings.get("max_frames")
                if max_frames == 0:
                    max_frames = None

                data = {
                    "sample_rate": settings.get("sample_rate", 1.0),
                    "max_frames": max_frames,
                    "fake_threshold": settings.get("fake_threshold", 0.7),
                    "suspicious_threshold": settings.get("suspicious_threshold", 0.4),
                }

                response = requests.post(url, files=files, data=data, timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
                return response.json()

        except requests.exceptions.ConnectionError:
            st.error(f"❌ Connection Error: Could not reach API at {self.base_url}. Is the backend running?")
            return None
        except requests.exceptions.HTTPError as e:
            detail = "Unknown error"
            try:
                detail = e.response.json().get("detail", str(e))
            except json.JSONDecodeError:
                detail = e.response.text
            st.error(f"❌ API Error: {detail}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")
            return None


def inject_custom_css():
    """Injects custom CSS for result styling."""
    st.markdown(
        """
        <style>
            .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
            .verdict-box {
                text-align: center; margin: 2rem 0; padding: 1rem; border-radius: 12px;
                color: white; font-weight: 800; font-size: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .real { background: linear-gradient(135deg, #00c853, #69f0ae); }
            .fake { background: linear-gradient(135deg, #d50000, #ff5252); }
            .suspicious { background: linear-gradient(135deg, #ff6d00, #ffd180); }
            .inconclusive { background: linear-gradient(135deg, #455a64, #90a4ae); }
            
            .metric-card {
                background: rgba(255, 255, 255, 0.05);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_results(result: Dict[str, Any]):
    """Renders the analysis results in a structured format."""
    st.divider()
    
    verdict = result.get("verdict", "INCONCLUSIVE")
    confidence = result.get("confidence", 0.0)
    
    # Verdict Banner
    css_class = verdict.lower()
    icon = {"REAL": "✅", "FAKE": "🚨", "SUSPICIOUS": "⚠️", "INCONCLUSIVE": "❓"}.get(verdict, "❓")
    
    st.markdown(
        f'<div class="verdict-box {css_class}">{icon} {verdict}</div>',
        unsafe_allow_html=True
    )

    # Detailed Metrics
    st.markdown(f"### Confidence: **{confidence:.1%}**")
    st.progress(confidence)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Duration", f"{result.get('duration_seconds', 0):.1f}s")
    with c2: st.metric("Analyzed Frames", result.get("frames_analyzed", 0))
    with c3: st.metric("Faces Found", result.get("frames_with_faces", 0))
    with c4: st.metric("Avg Fake Score", f"{result.get('average_fake_score', 0):.1%}")

    # Explanation Section
    st.markdown("### 📝 Analysis Report")
    st.info(result.get("verdict_text", "No detailed text available."))

    with st.expander("🔬 Technical Explanation"):
        st.write(result.get("explanation", "No technical explanation provided."))

    if result.get("recommendation"):
        st.markdown("### 💡 Recommendation")
        st.warning(result.get("recommendation"))

    # Download
    st.download_button(
        label="📥 Download JSON Report",
        data=json.dumps(result, indent=2),
        file_name=f"report_{result.get('video_path', 'video')}.json",
        mime="application/json",
    )


def sidebar_settings() -> Dict[str, Any]:
    """Renders sidebar and returns settings dict."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        st.subheader("Analysis Parameters")
        settings = {
            "sample_rate": st.slider("Sampling Rate (FPS)", 0.5, 5.0, 1.0, 0.5, help="Higher FPS = slower but more accurate"),
            "max_frames": st.number_input("Max Frames", 0, 500, 0, help="0 for unlimited"),
            "fake_threshold": st.slider("Fake Threshold", 0.50, 0.99, 0.70),
            "suspicious_threshold": st.slider("Suspicious Threshold", 0.10, 0.60, 0.40),
        }
        
        st.divider()
        st.markdown("### About")
        st.caption("Agentic Deepfake Classifier v1.0")
        st.caption("Running in HTTP Client Mode")
        
        return settings


def main():
    st.set_page_config(page_title="Deepfake Detective", page_icon="🕵️‍♂️", layout="wide")
    inject_custom_css()
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("# 🕵️‍♂️")
    with col2:
        st.title("Deepfake Detective")
        st.markdown("#### Autonomous AI Video Authentication Agent")

    client = DeepfakeClient(API_BASE_URL)
    settings = sidebar_settings()

    st.divider()
    
    uploaded_file = st.file_uploader("Upload Video for Analysis", type=["mp4", "mov", "avi", "webm"])

    if uploaded_file:
        # Save temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        col_video, col_action = st.columns([1, 1])
        with col_video:
            st.video(tmp_path)
            st.caption(f"Filename: {uploaded_file.name}")

        with col_action:
            st.markdown("### Ready to Analyze")
            st.markdown("The agent will extract frames, detect faces, and enable its reasoning engine to determine authenticity.")
            
            if st.button("🚀 Start Investigation"):
                with st.spinner("🕵️ Agent is analyzing video content..."):
                    # Simulate stages for UX
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Connecting to Cognitive Engine...")
                    progress_bar.progress(10)
                    time.sleep(0.5)
                    
                    status_text.text("Uploading Video Stream...")
                    progress_bar.progress(30)
                    
                    # Real API Call
                    result = client.analyze(tmp_path, settings)
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()

                if result:
                    display_results(result)
                
        # Cleanup
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    else:
        st.info("👆 Upload a supported video file to begin the investigation.")

if __name__ == "__main__":
    main()
