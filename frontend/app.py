"""
Agentic Deepfake Classifier - Streamlit Web Interface
"""

import streamlit as st
import sys
import os
import json
import tempfile
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
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
""", unsafe_allow_html=True)


def get_verdict_class(verdict: str) -> str:
    return {
        "REAL": "verdict-real",
        "FAKE": "verdict-fake",
        "SUSPICIOUS": "verdict-suspicious",
        "INCONCLUSIVE": "verdict-inconclusive"
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
            "suspicious_threshold": suspicious_threshold
        }


def render_results(result):
    st.markdown("---")
    st.subheader("📋 Results")
    
    verdict = result.verdict.value
    verdict_class = get_verdict_class(verdict)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f'<div style="text-align: center; margin: 2rem 0;">'
            f'<div class="{verdict_class}">{result.verdict.emoji} {verdict}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    st.markdown(f"### Confidence: {result.confidence:.1%}")
    st.progress(result.confidence)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frames Analyzed", result.frames_analyzed)
    with col2:
        st.metric("Faces Detected", result.frames_with_faces)
    with col3:
        st.metric("Avg Fake Score", f"{result.average_fake_score:.1%}")
    with col4:
        st.metric("Duration", f"{result.duration_seconds:.1f}s")
    
    st.markdown("### 📝 Explanation")
    st.info(result.verdict_text)
    
    with st.expander("🔬 Technical Details"):
        st.markdown(result.explanation)
    
    st.markdown("### 💡 Recommendation")
    st.warning(result.recommendation)
    
    st.markdown("---")
    st.download_button(
        label="📥 Download Report (JSON)",
        data=json.dumps(result.to_dict(), indent=2),
        file_name="deepfake_report.json",
        mime="application/json"
    )


def analyze_video(video_path: str, settings: dict):
    try:
        from src import DeepfakeAnalyzer
        
        analyzer = DeepfakeAnalyzer(
            sample_rate=settings["sample_rate"],
            max_frames=settings["max_frames"],
            fake_threshold=settings["fake_threshold"],
            suspicious_threshold=settings["suspicious_threshold"]
        )
        
        return analyzer.analyze(video_path, show_progress=False)
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None


def main():
    render_header()
    settings = render_sidebar()
    
    st.subheader("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm']
    )
    
    if uploaded_file:
        st.video(uploaded_file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                progress = st.progress(0)
                status = st.empty()
                
                status.text("Loading model...")
                progress.progress(20)
                
                result = analyze_video(tmp_path, settings)
                
                progress.progress(100)
                status.text("Complete!")
                time.sleep(0.3)
                progress.empty()
                status.empty()
                
                if result:
                    render_results(result)
        
        try:
            os.unlink(tmp_path)
        except:
            pass
    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; 
            background: rgba(255,255,255,0.02); border-radius: 16px;
            border: 2px dashed rgba(255,255,255,0.1); margin: 2rem 0;">
            <p style="font-size: 4rem; margin: 0;">📹</p>
            <p style="color: rgba(255,255,255,0.6);">Upload a video to begin</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
