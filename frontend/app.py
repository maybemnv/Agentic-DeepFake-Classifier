"""
Agentic Deepfake Classifier - Streamlit Web Interface
Beautiful and intuitive UI for deepfake video analysis.

Run with: streamlit run frontend/app.py
"""

import streamlit as st
import sys
import os
import json
import tempfile
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Card styling */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    /* Verdict badges */
    .verdict-real {
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.5rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
    }
    
    .verdict-fake {
        background: linear-gradient(135deg, #ff1744 0%, #ff5252 100%);
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.5rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 23, 68, 0.4);
    }
    
    .verdict-suspicious {
        background: linear-gradient(135deg, #ff9100 0%, #ffab40 100%);
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.5rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 145, 0, 0.4);
    }
    
    .verdict-inconclusive {
        background: linear-gradient(135deg, #78909c 0%, #90a4ae 100%);
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.5rem;
        display: inline-block;
    }
    
    /* Confidence meter */
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


def get_verdict_class(verdict: str) -> str:
    """Get CSS class for verdict badge."""
    classes = {
        "REAL": "verdict-real",
        "FAKE": "verdict-fake",
        "SUSPICIOUS": "verdict-suspicious",
        "INCONCLUSIVE": "verdict-inconclusive"
    }
    return classes.get(verdict, "verdict-inconclusive")


def render_header():
    """Render the header section."""
    st.markdown('<h1 class="main-header">🔍 Agentic Deepfake Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: rgba(255,255,255,0.7); font-size: 1.1rem;">'
        'Autonomous AI-powered video authenticity verification'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with settings."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Analysis Options")
        sample_rate = st.slider(
            "Frame Sample Rate (fps)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="Higher = more frames analyzed, slower but more accurate"
        )
        
        max_frames = st.number_input(
            "Max Frames (0 = unlimited)",
            min_value=0,
            max_value=100,
            value=0,
            help="Limit frames for faster analysis"
        )
        
        st.subheader("Decision Thresholds")
        fake_threshold = st.slider(
            "Fake Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Score >= this = FAKE"
        )
        
        suspicious_threshold = st.slider(
            "Suspicious Threshold",
            min_value=0.2,
            max_value=0.6,
            value=0.4,
            step=0.05,
            help="Score >= this (and < fake) = SUSPICIOUS"
        )
        
        st.markdown("---")
        
        st.subheader("📊 About")
        st.info(
            "This tool uses XceptionNet trained on the FaceForensics++ dataset "
            "to detect deepfake videos. Upload a video to get started!"
        )
        
        return {
            "sample_rate": sample_rate,
            "max_frames": max_frames if max_frames > 0 else None,
            "fake_threshold": fake_threshold,
            "suspicious_threshold": suspicious_threshold
        }


def render_upload_section():
    """Render the video upload section."""
    st.subheader("📤 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Supported formats: MP4, AVI, MOV, MKV, WebM"
    )
    
    return uploaded_file


def render_results(result):
    """Render analysis results."""
    st.markdown("---")
    st.subheader("📋 Analysis Results")
    
    # Verdict banner
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
    
    # Confidence
    st.markdown(f"### Confidence: {result.confidence:.1%}")
    st.progress(result.confidence)
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Frames Analyzed", result.frames_analyzed)
    
    with col2:
        st.metric("Faces Detected", result.frames_with_faces)
    
    with col3:
        st.metric("Avg Fake Score", f"{result.average_fake_score:.1%}")
    
    with col4:
        st.metric("Duration", f"{result.duration_seconds:.1f}s")
    
    # Explanation
    st.markdown("### 📝 Explanation")
    st.info(result.verdict_text)
    
    # Technical details expander
    with st.expander("🔬 Technical Details"):
        st.markdown(result.explanation)
    
    # Recommendation
    st.markdown("### 💡 Recommendation")
    st.warning(result.recommendation)
    
    # Download results
    st.markdown("---")
    result_json = json.dumps(result.to_dict(), indent=2)
    st.download_button(
        label="📥 Download Full Report (JSON)",
        data=result_json,
        file_name="deepfake_analysis_report.json",
        mime="application/json"
    )


def analyze_video(video_path: str, settings: dict):
    """Run analysis on uploaded video."""
    try:
        from src.analyzer import DeepfakeAnalyzer
        
        analyzer = DeepfakeAnalyzer(
            sample_rate=settings["sample_rate"],
            max_frames=settings["max_frames"],
            fake_threshold=settings["fake_threshold"],
            suspicious_threshold=settings["suspicious_threshold"]
        )
        
        result = analyzer.analyze(video_path, show_progress=False)
        return result
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None


def main():
    """Main application."""
    render_header()
    settings = render_sidebar()
    
    # Main content
    uploaded_file = render_upload_section()
    
    if uploaded_file is not None:
        # Show video preview
        st.video(uploaded_file)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Analyze button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing video... This may take a moment."):
                # Progress simulation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Loading model...")
                progress_bar.progress(20)
                
                result = analyze_video(tmp_path, settings)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                if result:
                    render_results(result)
        
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    else:
        # Show placeholder
        st.markdown(
            """
            <div style="
                text-align: center; 
                padding: 4rem 2rem; 
                background: rgba(255,255,255,0.02); 
                border-radius: 16px;
                border: 2px dashed rgba(255,255,255,0.1);
                margin: 2rem 0;
            ">
                <p style="font-size: 4rem; margin: 0;">📹</p>
                <p style="font-size: 1.2rem; color: rgba(255,255,255,0.6);">
                    Upload a video to begin analysis
                </p>
                <p style="font-size: 0.9rem; color: rgba(255,255,255,0.4);">
                    Supports MP4, AVI, MOV, MKV, WebM
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
