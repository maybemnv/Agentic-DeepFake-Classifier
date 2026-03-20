"""
Agentic Deepfake Classifier — Streamlit Web Interface
Supports single-video analysis and side-by-side comparison mode.
"""

from __future__ import annotations

import os
import json
import time
import tempfile
import requests
import streamlit as st
from typing import Any

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 300

# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------


class DeepfakeClient:
    """HTTP client for the Deepfake Analysis API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, **kwargs) -> dict[str, Any] | None:
        try:
            resp = requests.post(f"{self.base_url}{path}", timeout=TIMEOUT_SECONDS, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach API at {self.base_url}. Is the backend running?")
        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = e.response.text
            st.error(f"API Error: {detail}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
        return None

    def analyze(self, video_path: str, settings: dict[str, Any]) -> dict[str, Any] | None:
        max_frames = settings.get("max_frames") or None
        with open(video_path, "rb") as f:
            return self._post(
                "/analyze",
                files={"file": (os.path.basename(video_path), f, "video/mp4")},
                data={
                    "sample_rate": settings.get("sample_rate", 1.0),
                    "max_frames": max_frames,
                    "fake_threshold": settings.get("fake_threshold", 0.7),
                    "suspicious_threshold": settings.get("suspicious_threshold", 0.4),
                },
            )

    def compare(
        self,
        path1: str,
        path2: str,
        desc1: str = "Original",
        desc2: str = "Suspected deepfake",
    ) -> dict[str, Any] | None:
        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            return self._post(
                "/analyze/compare",
                files={
                    "video1": (os.path.basename(path1), f1, "video/mp4"),
                    "video2": (os.path.basename(path2), f2, "video/mp4"),
                },
                data={"video1_description": desc1, "video2_description": desc2},
            )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_LIGHT_CSS = """
<style>
:root {
    --bg: #f8f9fa;
    --card: #ffffff;
    --text: #212529;
    --subtext: #6c757d;
    --border: #dee2e6;
    --accent: #0d6efd;
}
"""

_DARK_CSS = """
<style>
:root {
    --bg: #121212;
    --card: #1e1e1e;
    --text: #e0e0e0;
    --subtext: #9e9e9e;
    --border: #333333;
    --accent: #4ea8ff;
}
"""

_SHARED_CSS = """
body { background-color: var(--bg); color: var(--text); }
.block-container { padding-top: 1.5rem; }

.verdict-container {
    display: flex; align-items: center; justify-content: center;
    margin: 1.5rem 0; padding: 1.25rem; border-radius: 8px;
    background: var(--card); border: 1px solid var(--border);
}
.verdict-label { font-size: 1.1rem; font-weight: 500; color: var(--subtext); margin-right: 1rem; }
.verdict-badge {
    font-size: 1.6rem; font-weight: 700; padding: 0.4rem 1.2rem;
    border-radius: 6px; text-transform: uppercase; letter-spacing: 1px;
}
.real    { background: #198754; color: #fff; }
.fake    { background: #dc3545; color: #fff; }
.suspicious { background: #ffc107; color: #212529; }
.inconclusive { background: #6c757d; color: #fff; }

.metric-box {
    background: var(--card); border: 1px solid var(--border); border-radius: 6px;
    padding: 0.9rem; text-align: center; margin-bottom: 0.5rem;
}
.metric-value { font-size: 1.4rem; font-weight: 600; color: var(--text); }
.metric-label { font-size: 0.8rem; color: var(--subtext); text-transform: uppercase; letter-spacing: 0.5px; }

.stButton>button {
    width: 100%; border-radius: 6px; height: 3em; font-weight: 600;
    background-color: var(--accent); color: #fff; border: none;
}
</style>
"""


def inject_css(dark: bool) -> None:
    base = _DARK_CSS if dark else _LIGHT_CSS
    st.markdown(base + _SHARED_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def verdict_html(verdict: str) -> str:
    css = verdict.lower()
    return f"""
    <div class="verdict-container">
        <span class="verdict-label">Verdict:</span>
        <span class="verdict-badge {css}">{verdict}</span>
    </div>"""


def metric_box(value: str, label: str) -> str:
    return f"""
    <div class="metric-box">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def _show_result(result: dict[str, Any], title: str = "Analysis Result") -> None:
    st.markdown(f"### {title}")
    st.markdown(verdict_html(result.get("verdict", "INCONCLUSIVE")), unsafe_allow_html=True)

    conf = result.get("confidence", 0.0)
    st.markdown(f"**Confidence:** {conf:.1%}")
    st.progress(conf)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_box(f"{result.get('duration_seconds', 0):.1f}s", "Duration"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_box(str(result.get("frames_analyzed", 0)), "Frames Analyzed"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_box(str(result.get("frames_with_faces", 0)), "Faces Detected"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_box(f"{result.get('average_fake_score', 0):.1%}", "Avg Fake Score"),
            unsafe_allow_html=True,
        )

    st.info(result.get("verdict_text", ""))

    with st.expander("Technical Explanation"):
        st.markdown(result.get("explanation", ""))

    if result.get("recommendation"):
        st.warning(result["recommendation"])

    frame_scores = result.get("frame_scores") or []
    if frame_scores:
        with st.expander("Frame-by-Frame Fake Score Chart"):
            import pandas as pd

            df = pd.DataFrame({"Frame": range(len(frame_scores)), "Fake Score": frame_scores})
            st.bar_chart(df.set_index("Frame"))
            st.caption("Each bar = one analysed frame. Higher = more likely manipulated.")

    quality = result.get("quality_metrics")
    if quality:
        with st.expander("Video Quality Metrics"):
            qcols = st.columns(4)
            labels = ["Resolution", "Compression", "Lighting", "Face Clarity"]
            keys = ["resolution_score", "compression_score", "lighting_score", "face_clarity_score"]
            for col, label, key in zip(qcols, labels, keys):
                with col:
                    st.metric(label, f"{quality.get(key, 0):.0%}")
            if quality.get("issues"):
                st.warning("Issues: " + ", ".join(quality["issues"]))
            if quality.get("recommendations"):
                st.info("Recommendations: " + "; ".join(quality["recommendations"]))

    st.download_button(
        "Download Report (JSON)",
        data=json.dumps(result, indent=2, default=str),
        file_name=f"report_{int(time.time())}.json",
        mime="application/json",
    )


def _show_comparison(result: dict[str, Any], desc1: str, desc2: str) -> None:
    st.markdown("### Comparison Result")

    col1, col2 = st.columns(2)
    with col1:
        _show_result(result["video1_result"], title=desc1)
    with col2:
        _show_result(result["video2_result"], title=desc2)

    st.divider()
    st.markdown("### Differential Analysis")
    st.info(result.get("conclusion", ""))
    st.code(result.get("differential_analysis", ""), language=None)

    scores1 = result.get("frame_scores_video1") or []
    scores2 = result.get("frame_scores_video2") or []
    if scores1 or scores2:
        with st.expander("Score Heatmap — Frame Comparison"):
            import pandas as pd

            max_len = max(len(scores1), len(scores2))
            # Pad shorter list
            s1 = scores1 + [None] * (max_len - len(scores1))
            s2 = scores2 + [None] * (max_len - len(scores2))
            df = pd.DataFrame({desc1: s1, desc2: s2})
            st.line_chart(df)
            st.caption("Per-frame fake probability for both videos overlaid.")

    st.metric("Similarity Score", f"{result.get('similarity_score', 0):.1%}")


# ---------------------------------------------------------------------------
# Sidebar / Settings
# ---------------------------------------------------------------------------


def sidebar_settings() -> tuple[dict[str, Any], bool, str]:
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        dark_mode = st.toggle("Dark Mode", value=False)

        st.divider()
        mode = st.radio("Analysis Mode", ["Single Video", "Comparison"])

        st.markdown("#### Parameters")
        params = {
            "sample_rate": st.slider("Sampling Rate (FPS)", 0.5, 5.0, 1.0, 0.5),
            "max_frames": st.number_input("Max Frames (0 = unlimited)", 0, 500, 0),
            "fake_threshold": st.slider("Fake Threshold", 0.50, 0.99, 0.70),
            "suspicious_threshold": st.slider("Suspicious Threshold", 0.10, 0.60, 0.40),
        }

        st.divider()
        st.caption("Agentic Deepfake Classifier v1.1.0")
        # Grad-CAM note — requires server-side pipeline support not yet wired
        st.caption(
            "ℹ️ Grad-CAM overlay video export is planned for a future release "
            "and requires dedicated server-side heatmap generation in the detection pipeline."
        )

    return params, dark_mode, mode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Deepfake Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    params, dark_mode, mode = sidebar_settings()
    inject_css(dark_mode)

    st.title("Deepfake Analysis Dashboard")
    st.markdown("Autonomous Video Authenticity Verification System")
    st.divider()

    client = DeepfakeClient(API_BASE_URL)

    if mode == "Single Video":
        uploaded = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "webm", "mkv"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            col_v, col_a = st.columns([1, 1])
            with col_v:
                st.video(tmp_path)
                st.caption(uploaded.name)
            with col_a:
                st.markdown("### Ready")
                st.markdown("Click below to run the full agentic analysis pipeline.")
                if st.button("Run Analysis"):
                    with st.spinner("Analysing video…"):
                        prog = st.progress(0)
                        prog.progress(20)
                        result = client.analyze(tmp_path, params)
                        prog.progress(100)
                        prog.empty()
                    if result:
                        _show_result(result)

            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        else:
            st.info("Upload a video file to begin.")

    else:  # Comparison mode
        st.markdown("### Upload two videos to compare side-by-side")
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.file_uploader(
                "Video 1 (Original)", type=["mp4", "mov", "avi", "webm", "mkv"], key="v1"
            )
            desc1 = st.text_input("Label", value="Original", key="d1")
        with col2:
            f2 = st.file_uploader(
                "Video 2 (Suspected Deepfake)", type=["mp4", "mov", "avi", "webm", "mkv"], key="v2"
            )
            desc2 = st.text_input("Label", value="Suspected Deepfake", key="d2")

        if f1 and f2:
            paths: list[str] = []
            for f in [f1, f2]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(f.read())
                    paths.append(tmp.name)

            if st.button("Run Comparison"):
                with st.spinner("Comparing videos…"):
                    result = client.compare(paths[0], paths[1], desc1, desc2)
                if result:
                    _show_comparison(result, desc1, desc2)

            for p in paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        else:
            st.info("Upload both videos to enable comparison.")


if __name__ == "__main__":
    main()
