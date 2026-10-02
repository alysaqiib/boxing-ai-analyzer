"""
Boxing AI Performance Analyzer -- Web App
----------------------------------------------
A simple local web interface around the pipeline: upload a video,
click a button, watch it process, then view the annotated video and
stats right in your browser.

RUN THIS WITH:
    streamlit run src/app.py

This does NOT reimplement any analysis logic -- it just wraps
run_pipeline.py's run_full_pipeline() function in a UI.
"""

import sys
import time
import shutil
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # so imports below work when run via `streamlit run`

from run_pipeline import run_full_pipeline

st.set_page_config(page_title="Boxing AI Performance Analyzer", page_icon="🥊", layout="wide")

UPLOAD_DIR = Path("data/videos")
OUTPUT_DIR = Path("data/output")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.title("🥊 Boxing AI Performance Analyzer")
st.caption("Upload a boxing/sparring video and get automatic fighter tracking, "
           "strike classification, defensive action detection, and footwork stats.")

st.divider()

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    video_path = UPLOAD_DIR / uploaded_file.name
    with open(video_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    st.video(str(video_path))

    if st.button("Run Analysis", type="primary"):
        progress_bar = st.progress(0, text="Starting...")
        status_box = st.empty()

        def update_progress(step, total, message):
            progress_bar.progress(step / total, text=message.split("\n")[0])
            status_box.text(message)

        start_time = time.time()
        with st.spinner("Running full pipeline -- this can take a few minutes depending on video length..."):
            try:
                result = run_full_pipeline(
                    video_path=str(video_path),
                    out_dir=str(OUTPUT_DIR),
                    progress_callback=update_progress,
                )
            except Exception as e:
                st.error(f"Something went wrong during processing: {e}")
                st.stop()

        elapsed = time.time() - start_time
        progress_bar.progress(1.0, text="Done!")
        st.success(f"Analysis complete in {elapsed:.0f} seconds.")

        st.divider()
        st.subheader("Annotated Video")
        st.video(result["video"])

        st.divider()
        st.subheader("Final Stats")
        st.image(result["summary_png"])

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Per-Fighter Summary")
            st.caption("Note: 'distance_covered_relative_units' is a relative body-length "
                       "measure, not real-world meters/feet -- only meaningful for comparing "
                       "the two fighters within this same video.")
            summary_df = pd.read_csv(result["summary_csv"])
            st.dataframe(summary_df, use_container_width=True)
            st.download_button(
                "Download summary CSV", data=Path(result["summary_csv"]).read_bytes(),
                file_name=Path(result["summary_csv"]).name, mime="text/csv",
            )
        with col2:
            st.subheader("Every Detected Strike")
            strikes_df = pd.read_csv(result["strikes_csv"])
            st.dataframe(strikes_df, use_container_width=True, height=300)
            st.download_button(
                "Download strikes CSV", data=Path(result["strikes_csv"]).read_bytes(),
                file_name=Path(result["strikes_csv"]).name, mime="text/csv",
            )

        st.divider()
        st.download_button(
            "Download annotated video", data=Path(result["video"]).read_bytes(),
            file_name=Path(result["video"]).name, mime="video/mp4",
        )
else:
    st.info("Upload a video above to get started. Supported formats: MP4, MOV, AVI, MKV.")
