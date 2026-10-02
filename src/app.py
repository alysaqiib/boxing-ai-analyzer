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

import cv2
import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # so imports below work when run via `streamlit run`

from run_pipeline import run_full_pipeline, rebuild_corrected_reports

st.set_page_config(page_title="Boxing AI Performance Analyzer", page_icon="🥊", layout="wide")

UPLOAD_DIR = Path("data/videos")
OUTPUT_DIR = Path("data/output")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_event_clip(video_path, start_frame, end_frame, output_path, playback_rate=0.5):
    """Write a short slow-motion clip around one detected event."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Could not read the source video's dimensions.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps * playback_rate, 1.0),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the event preview video.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    frame_idx = max(0, start_frame)
    written = 0
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1
        frame_idx += 1
    cap.release()
    writer.release()
    if written == 0:
        raise ValueError("The selected event has no readable video frames.")


st.title("🥊 Boxing AI Performance Analyzer")
st.caption("Upload a boxing/sparring video and get automatic fighter tracking, "
           "strike classification, defensive action detection, and footwork stats.")

st.divider()

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
round_duration_seconds = st.number_input(
    "Round length (seconds)", min_value=10.0, max_value=1800.0,
    value=180.0, step=10.0,
    help="Analytics are grouped into consecutive time windows of this length.",
)

if uploaded_file is not None:
    video_path = UPLOAD_DIR / uploaded_file.name
    with open(video_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    st.video(str(video_path))

    run_clicked = st.button("Run Analysis", type="primary")
    has_saved_analysis = (
        st.session_state.get("analysis_result") is not None
        and st.session_state.get("analysis_video_path") == str(video_path)
    )
    if run_clicked or has_saved_analysis:
        progress_bar = st.progress(0, text="Starting...")
        status_box = st.empty()

        if run_clicked:
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
                        round_duration_seconds=round_duration_seconds,
                    )
                except Exception as e:
                    st.error(f"Something went wrong during processing: {e}")
                    st.stop()
            st.session_state["analysis_result"] = result
            st.session_state["analysis_video_path"] = str(video_path)
            elapsed = time.time() - start_time
            progress_bar.progress(1.0, text="Done!")
            st.success(f"Analysis complete in {elapsed:.0f} seconds.")
        else:
            result = st.session_state["analysis_result"]
            video_path = Path(st.session_state["analysis_video_path"])
            progress_bar.empty()
            status_box.empty()

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
            st.caption("Confidence is an estimate based on motion strength and event resolution.")
            editable_columns = ["punch_type", "result", "target_zone"]
            edited_strikes_df = st.data_editor(
                strikes_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "punch_type": st.column_config.SelectboxColumn(
                        "Punch type",
                        options=["Jab/Cross", "Hook", "Uppercut"],
                    ),
                    "result": st.column_config.SelectboxColumn(
                        "Result",
                        options=["Landed", "Blocked", "Missed", "Unknown"],
                    ),
                    "target_zone": st.column_config.SelectboxColumn(
                        "Target zone",
                        options=["Head", "Body", "Leg", ""],
                    ),
                },
                disabled=[column for column in strikes_df.columns if column not in editable_columns],
                key="strike_corrections",
            )
            if st.button("Save corrected strike data"):
                corrected_path = OUTPUT_DIR / f"{video_path.stem}_strikes_corrected.csv"
                edited_strikes_df.to_csv(corrected_path, index=False)
                try:
                    corrected_result = rebuild_corrected_reports(
                        edited_strikes_df.to_dict("records"),
                        result,
                        str(OUTPUT_DIR),
                        video_path.stem,
                        float(round_duration_seconds),
                    )
                except (TypeError, ValueError, OSError) as error:
                    st.error(f"Could not rebuild corrected analytics: {error}")
                else:
                    result.update(corrected_result)
                    st.session_state["analysis_result"] = result
                st.session_state["corrected_strikes_path"] = str(corrected_path)
                st.success(
                    "Corrections saved. Summary, rounds, combinations, performance, "
                    "and dashboard data were rebuilt. The original analysis video is unchanged."
                )
            if st.session_state.get("corrected_strikes_path") and Path(
                st.session_state["corrected_strikes_path"]
            ).exists():
                corrected_path = Path(st.session_state["corrected_strikes_path"])
                st.download_button(
                    "Download corrected strikes CSV",
                    data=corrected_path.read_bytes(),
                    file_name=corrected_path.name,
                    mime="text/csv",
                )

        st.divider()
        st.subheader("Round-by-Round Analytics")
        round_df = pd.read_csv(result["round_summary_csv"])
        st.dataframe(round_df, use_container_width=True)
        st.download_button(
            "Download round summary CSV", data=Path(result["round_summary_csv"]).read_bytes(),
            file_name=Path(result["round_summary_csv"]).name, mime="text/csv",
        )

        st.divider()
        st.subheader("Interactive Dashboard")
        dashboard_col1, dashboard_col2 = st.columns(2)
        with dashboard_col1:
            st.caption("Punch volume and outcomes")
            comparison = summary_df.set_index("fighter")[
                ["total_strikes", "landed", "blocked", "missed"]
            ]
            st.bar_chart(comparison)

            st.caption("Punch type distribution")
            punch_types = summary_df.set_index("fighter")[["jab_cross", "hook", "uppercut"]]
            st.bar_chart(punch_types)

        with dashboard_col2:
            st.caption("Accuracy by round")
            accuracy_by_round = round_df.pivot(
                index="round", columns="fighter", values="accuracy_percent"
            ).fillna(0)
            st.line_chart(accuracy_by_round)

            st.caption("Target zones for landed punches")
            target_zones = summary_df.set_index("fighter")[
                ["head_targets", "body_targets", "leg_targets"]
            ]
            st.bar_chart(target_zones)

        st.divider()
        st.subheader("Performance Score & Coaching Feedback")
        performance_df = pd.read_csv(result["performance_csv"])
        st.dataframe(performance_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download performance report",
            data=Path(result["performance_csv"]).read_bytes(),
            file_name=Path(result["performance_csv"]).name,
            mime="text/csv",
        )

        st.divider()
        st.subheader("Event Review Timeline")
        event_rows = []
        for _, strike in strikes_df.iterrows():
            event_rows.append({
                "label": (
                    f"{strike['time_seconds']:.2f}s - {strike['fighter']} "
                    f"{strike['punch_type']} ({strike['result']})"
                ),
                "frame": int(strike["frame"]),
            })
        fps = result.get("fps", 30.0)
        for identity, defense in result["defensive_by_identity"].items():
            for event in defense["duck_slip_events"]:
                event_rows.append({
                    "label": (
                        f"{event['frame_idx'] / fps:.2f}s - Fighter {identity} "
                        f"{event['action_type']}"
                    ),
                    "frame": event["frame_idx"],
                })
        event_rows.sort(key=lambda event: event["frame"])

        if event_rows:
            selected_label = st.selectbox(
                "Select an event to review", [event["label"] for event in event_rows]
            )
            selected_event = next(event for event in event_rows if event["label"] == selected_label)
            clip_path = OUTPUT_DIR / f"{video_path.stem}_event_preview.mp4"
            if st.button("Create slow-motion preview"):
                half_window = int(fps * 2)
                try:
                    extract_event_clip(
                        video_path, selected_event["frame"] - half_window,
                        selected_event["frame"] + half_window, clip_path,
                    )
                    st.session_state["event_clip_path"] = str(clip_path)
                    st.session_state["event_clip_label"] = selected_label
                except (OSError, RuntimeError, ValueError) as error:
                    st.error(f"Could not create event preview: {error}")
            if (st.session_state.get("event_clip_label") == selected_label
                    and st.session_state.get("event_clip_path")
                    and Path(
                st.session_state["event_clip_path"]
            ).exists()):
                st.video(st.session_state["event_clip_path"])
                st.download_button(
                    "Download event preview",
                    data=Path(st.session_state["event_clip_path"]).read_bytes(),
                    file_name=Path(st.session_state["event_clip_path"]).name,
                    mime="video/mp4",
                )
        else:
            st.info("No reviewable events were detected in this video.")

        st.divider()
        st.subheader("Detected Combinations")
        combinations_df = pd.read_csv(result["combinations_csv"])
        if combinations_df.empty:
            st.info("No two-or-more-punch combinations were detected.")
        else:
            st.dataframe(combinations_df, use_container_width=True)
            st.download_button(
                "Download combinations CSV",
                data=Path(result["combinations_csv"]).read_bytes(),
                file_name=Path(result["combinations_csv"]).name,
                mime="text/csv",
            )

        st.divider()
        st.download_button(
            "Download annotated video", data=Path(result["video"]).read_bytes(),
            file_name=Path(result["video"]).name, mime="video/mp4",
        )
else:
    st.info("Upload a video above to get started. Supported formats: MP4, MOV, AVI, MKV.")
