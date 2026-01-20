"""
Agentic Deepfake Classifier - Streamlit Web Interface
HTTP Client Only - Does NOT import torch, classifier, or xception.
Calls the FastAPI backend for all analysis.
"""

import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import time

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .verdict-real {
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
        color: white; padding: 0.5rem 2rem; border-radius: 50px;
        font-weight: 700; font-size: 1.5rem; display: inline-block;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #ff1744 0%, #ff5252 100%);
        color: white; padding: 0.5rem 2rem; border-radius: 50px;
        font-weight: 700; font-size: 1.5rem; display: inline-block;
    }
    .verdict-suspicious {
        background: linear-gradient(135deg, #ff9100 0%, #ffab40 100%);
        color: white; padding: 0.5rem 2rem; border-radius: 50px;
        font-weight: 700; font-size: 1.5rem; display: inline-block;
    }
    .verdict-inconclusive {
        background: linear-gradient(135deg, #78909c 0%, #90a4ae 100%);
        color: white; padding: 0.5rem 2rem; border-radius: 50px;
        font-weight: 700; font-size: 1.5rem; display: inline-block;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_verdict_class(verdict: str) -> str:
    return {
        "REAL": "verdict-real",
        "FAKE": "verdict-fake",
        "SUSPICIOUS": "verdict-suspicious",
        "INCONCLUSIVE": "verdict-inconclusive",
    }.get(verdict, "verdict-inconclusive")


def render_header():
    st.title("🔍 Agentic Deepfake Detector")
    st.markdown("*Autonomous AI-powered video authenticity verification*")
    st.markdown("---")


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")

        sample_rate = st.slider("Frame Sample Rate (fps)", 0.5, 5.0, 1.0, 0.5)
        max_frames = st.number_input("Max Frames (0 = unlimited)", 0, 100, 0)
        fake_threshold = st.slider("Fake Threshold", 0.5, 0.9, 0.7, 0.05)
        suspicious_threshold = st.slider("Suspicious Threshold", 0.2, 0.6, 0.4, 0.05)

        st.markdown("---")
        st.info("Upload a video to analyze for deepfake manipulation.")

        return {
            "sample_rate": sample_rate,
            "max_frames": max_frames if max_frames > 0 else None,
            "fake_threshold": fake_threshold,
            "suspicious_threshold": suspicious_threshold,
        }


def render_results(result):
    st.markdown("---")
    st.subheader("📋 Results")

    verdict = result.get("verdict", "INCONCLUSIVE")
    verdict_class = get_verdict_class(verdict)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        verdict_emoji = {
            "REAL": "✅",
            "FAKE": "❌",
            "SUSPICIOUS": "⚠️",
            "INCONCLUSIVE": "❓",
        }.get(verdict, "❓")
        st.markdown(
            f'<div style="text-align: center; margin: 2rem 0;">'
            f'<div class="{verdict_class}">{verdict_emoji} {verdict}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    confidence = result.get("confidence", 0)
    st.markdown(f"### Confidence: {confidence:.1%}")
    st.progress(confidence)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frames Analyzed", result.get("frames_analyzed", 0))
    with col2:
        st.metric("Faces Detected", result.get("frames_with_faces", 0))
    with col3:
        st.metric("Avg Fake Score", f"{result.get('average_fake_score', 0):.1%}")
    with col4:
        st.metric("Duration", f"{result.get('duration_seconds', 0):.1f}s")

    st.markdown("### 📝 Explanation")
    st.info(result.get("verdict_text", ""))

    with st.expander("🔬 Technical Details"):
        st.markdown(result.get("explanation", ""))

    st.markdown("### 💡 Recommendation")
    st.warning(result.get("recommendation", ""))

    st.markdown("---")
    st.download_button(
        label="📥 Download Report (JSON)",
        data=json.dumps(result, indent=2),
        file_name="deepfake_report.json",
        mime="application/json",
    )


def analyze_video_http(video_path: str, settings: dict):
    """
    Analyze video via HTTP API - no torch/classifier imports.
    """
    import requests

    url = f"{API_BASE_URL}/analyze"

    with open(video_path, "rb") as f:
        files = {"file": (os.path.basename(video_path), f, "video/mp4")}
        data = {
            "sample_rate": settings["sample_rate"],
            "max_frames": settings["max_frames"],
            "fake_threshold": settings["fake_threshold"],
            "suspicious_threshold": settings["suspicious_threshold"],
        }

        try:
            response = requests.post(url, files=files, data=data, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to API at {API_BASE_URL}. Make sure the server is running."
            )
            return None
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.json().get('detail', str(e))}")
            return None
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None


def main():
    render_header()
    settings = render_sidebar()

    st.subheader("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov", "mkv", "webm"]
    )

    if uploaded_file:
        st.video(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                progress = st.progress(0)
                status = st.empty()

                status.text("Connecting to API...")
                progress.progress(10)

                progress.progress(30)
                status.text("Uploading video...")

                result = analyze_video_http(tmp_path, settings)

                if result:
                    progress.progress(100)
                    status.text("Complete!")
                    time.sleep(0.3)
                    progress.empty()
                    status.empty()

                    render_results(result)
                else:
                    progress.empty()
                    status.empty()

        try:
            os.unlink(tmp_path)
        except:
            pass
    else:
        st.markdown(
            """
        <div style="text-align: center; padding: 4rem 2rem;
            background: rgba(255,255,255,0.02); border-radius: 16px;
            border: 2px dashed rgba(255,255,255,0.1); margin: 2rem 0;">
            <p style="font-size: 4rem; margin: 0;">📹</p>
            <p style="color: rgba(255,255,255,0.6);">Upload a video to begin</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
