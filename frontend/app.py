"""
Agentic Deepfake Classifier - Streamlit Web Interface

Modular, professional, and HTTP-only client for the Deepfake Analysis API.
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
                
                # Handle max_frames=0 (Unlimited) from UI
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
            st.error(f"Connection Error: Could not reach API at {self.base_url}. Please ensure the backend service is running.")
            return None
        except requests.exceptions.HTTPError as e:
            detail = "Unknown error"
            try:
                detail = e.response.json().get("detail", str(e))
            except json.JSONDecodeError:
                detail = e.response.text
            st.error(f"API Error: {detail}")
            return None
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")
            return None


def inject_custom_css():
    """Injects professional CSS for result styling."""
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2rem; }
            .stButton>button { 
                width: 100%; 
                border-radius: 6px; 
                height: 3em; 
                font-weight: 600;
                background-color: #007bff;
                color: white;
            }
            .verdict-container {
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 2rem 0;
                padding: 1.5rem;
                border-radius: 8px;
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
            }
            .verdict-label {
                font-size: 1.2rem;
                font-weight: 500;
                color: #495057;
                margin-right: 1rem;
            }
            .verdict-badge {
                font-size: 1.8rem;
                font-weight: 700;
                padding: 0.5rem 1.5rem;
                border-radius: 6px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .real { background-color: #28a745; color: white; }
            .fake { background-color: #dc3545; color: white; }
            .suspicious { background-color: #ffc107; color: #212529; }
            .inconclusive { background-color: #6c757d; color: white; }
            
            .metric-container {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 1rem;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .metric-value {
                font-size: 1.5rem;
                font-weight: 600;
                color: #212529;
            }
            .metric-label {
                font-size: 0.875rem;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_results(result: Dict[str, Any]):
    """Renders the analysis results in a professional format."""
    st.divider()
    
    verdict = result.get("verdict", "INCONCLUSIVE")
    confidence = result.get("confidence", 0.0)
    
    # Verdict Section
    css_class = verdict.lower()
    
    st.markdown(
        f"""
        <div class="verdict-container">
            <span class="verdict-label">Analysis Verdict:</span>
            <span class="verdict-badge {css_class}">{verdict}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Detailed Metrics Grid
    st.markdown(f"#### Confidence Score: {confidence:.1%}")
    st.progress(confidence)

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{result.get('duration_seconds', 0):.1f}s</div>
                <div class="metric-label">Duration</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{result.get('frames_analyzed', 0)}</div>
                <div class="metric-label">Frames Analyzed</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{result.get('frames_with_faces', 0)}</div>
                <div class="metric-label">Faces Detected</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{result.get('average_fake_score', 0):.1%}</div>
                <div class="metric-label">Avg Manipulation Score</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("### Analysis Report")
    
    with st.container():
        st.markdown("#### Executive Summary")
        st.info(result.get("verdict_text", "No summary available."))

    with st.expander("Technical Details"):
        st.markdown(result.get("explanation", "No technical explanation provided."))

    if result.get("recommendation"):
        st.markdown("#### Recommendation")
        st.warning(result.get("recommendation"))

    # Download
    st.download_button(
        label="Download Full Report (JSON)",
        data=json.dumps(result, indent=2),
        file_name=f"analysis_report_{int(time.time())}.json",
        mime="application/json",
    )


def sidebar_settings() -> Dict[str, Any]:
    """Renders sidebar and returns settings dict."""
    with st.sidebar:
        st.markdown("### Configuration")
        
        st.markdown("#### Parameters")
        settings = {
            "sample_rate": st.slider("Sampling Rate (FPS)", 0.5, 5.0, 1.0, 0.5, help="Higher FPS increases accuracy but processing time."),
            "max_frames": st.number_input("Max Frames", 0, 500, 0, help="Set to 0 for unlimited frames."),
            "fake_threshold": st.slider("Fake Threshold", 0.50, 0.99, 0.70),
            "suspicious_threshold": st.slider("Suspicious Threshold", 0.10, 0.60, 0.40),
        }
        
        st.divider()
        st.markdown("#### System Info")
        st.text("Agentic Deepfake Classifier")
        st.text("Version: 1.0.0")
        st.text("Mode: Client-Server")
        
        return settings


def main():
    st.set_page_config(page_title="Deepfake Analysis Dashboard", page_icon=None, layout="wide")
    inject_custom_css()
    
    st.title("Deepfake Analysis Dashboard")
    st.markdown("Autonomous Video Authenticity Verification System")

    client = DeepfakeClient(API_BASE_URL)
    settings = sidebar_settings()

    st.divider()
    
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "webm", "mkv"])

    if uploaded_file:
        # Save temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        col_video, col_info = st.columns([1, 1])
        with col_video:
            st.video(tmp_path)
            st.caption(f"Source: {uploaded_file.name}")

        with col_info:
            st.markdown("### Analysis Status")
            st.markdown("Ready to initialize analysis pipeline.")
            
            if st.button("Initialize Analysis"):
                with st.spinner("Processing video stream..."):
                    # UX Progress Simulation
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Establishing secure connection...")
                    progress_bar.progress(10)
                    time.sleep(0.3)
                    
                    status_text.text("Transmitting video data...")
                    progress_bar.progress(30)
                    
                    # Execute Analysis
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
        st.info("Please upload a video file to begin analysis.")

if __name__ == "__main__":
    main()
