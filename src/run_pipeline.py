"""
Full Pipeline: video in -> fully annotated analytics video + stats out
--------------------------------------------------------------------------
Combines everything built so far into one command:
  1. Pose tracking (pose_tracker.py)
  2. Glove-confirmed fighter detection (glove_confirmed_tracker.py)
  3. Appearance-based re-identification (reid_fighter_tracker.py)
  4. Strike classification (strike_classifier.py)

Outputs (all named after your input video):
  <name>_analysis.mp4   -- single annotated video: skeleton + fighter ID
                            colors + strike labels + a stats card outro
  <name>_strikes.csv     -- every individual detected strike
  <name>_summary.csv     -- one row per fighter: totals by punch type
  <name>_summary.png     -- a simple side-by-side stats card image

This file does NOT reimplement any logic -- it imports and orchestrates
the functions already built and tested in the other Phase files.
"""

import cv2
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

from pose_tracker import SKELETON_EDGES
from glove_confirmed_tracker import run_glove_confirmed_tracking
from reid_fighter_tracker import (
    compute_appearance_signatures,
    cluster_into_two_identities,
    IDENTITY_COLORS,
)
from strike_classifier import (
    build_identity_frame_series,
    detect_strikes_for_identity,
    compute_adaptive_speed_threshold,
    LABEL_DISPLAY_FRAMES,
)
from strike_resolution import resolve_all_strikes, summarize_results
from defensive_actions import analyze_defensive_actions
from footwork_analysis import analyze_footwork
from live_panel import LivePanelState, render_frame_with_panel, estimate_positions_from_frame, PANEL_WIDTH

STATS_CARD_SECONDS = 3  # how long the outro stats card stays on screen


def _open_video_writer(out_path, fps, size):
    """Tries H.264 (avc1) first for broad compatibility (plays natively in
    Windows' built-in video app, browsers, phones, etc.) -- some systems
    lack this encoder, so falls back to mp4v (works everywhere via OpenCV
    itself, but Windows' native Movies & TV app often can't play it)."""
    for fourcc_code in ["avc1", "H264", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if writer.isOpened():
            print(f"    (video encoder: {fourcc_code})")
            return writer
        writer.release()
    raise RuntimeError(f"Could not open a video writer for {out_path} with any known codec.")


def _text_size(text, font_scale, thickness):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    return w, h


def _put_text(img, text, x, y, font_scale, color, thickness, centered=False, right_aligned=False):
    w, h = _text_size(text, font_scale, thickness)
    if centered:
        x = x - w // 2
    elif right_aligned:
        x = x - w
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return w, h


def build_stats_card(strikes_by_identity: dict, identity_stats: dict, footwork_by_identity: dict,
                      width: int, height: int) -> np.ndarray:
    """Builds a polished 'Final Stats' outro card -- two side-by-side panels,
    one per fighter, with color-matched accents and right-aligned numbers."""
    BG = (24, 22, 20)
    PANEL_BG = (38, 36, 34)
    MUTED = (150, 150, 150)
    WHITE = (245, 245, 245)

    card = np.full((height, width, 3), BG, dtype=np.uint8)

    # Title, centered, with a thin accent underline
    title_y = int(height * 0.11)
    _put_text(card, "FINAL STATS", width // 2, title_y, 1.6, WHITE, 3, centered=True)
    underline_w = 140
    cv2.line(card, (width // 2 - underline_w // 2, title_y + 18),
             (width // 2 + underline_w // 2, title_y + 18), MUTED, 2, cv2.LINE_AA)

    margin = int(width * 0.05)
    gap = int(width * 0.03)
    panel_w = (width - margin * 2 - gap) // 2
    panel_top = int(height * 0.18)
    panel_bottom = int(height * 0.94)

    panels = [
        {"identity": "A", "x": margin},
        {"identity": "B", "x": margin + panel_w + gap},
    ]

    for panel in panels:
        identity = panel["identity"]
        x = panel["x"]
        color = IDENTITY_COLORS.get(identity, (200, 200, 200))

        # Card background
        cv2.rectangle(card, (x, panel_top), (x + panel_w, panel_bottom), PANEL_BG, -1)
        # Color accent strip down the left edge of the panel
        cv2.rectangle(card, (x, panel_top), (x + 6, panel_bottom), color, -1)

        pad = 30
        cx = x + pad
        cy = panel_top + 55

        strikes = strikes_by_identity.get(identity, [])
        type_counts = defaultdict(int)
        result_counts = defaultdict(int)
        for s in strikes:
            type_counts[s["punch_type"]] += 1
            result_counts[s.get("result", "Unknown")] += 1

        # Fighter name
        _put_text(card, f"FIGHTER {identity}", cx, cy, 1.15, color, 2)
        cy += 20
        cv2.line(card, (cx, cy), (x + panel_w - pad, cy), (60, 58, 55), 1, cv2.LINE_AA)
        cy += 45

        def stat_row(label, value, label_color=WHITE, value_color=WHITE, scale=0.72):
            _put_text(card, label, cx, cy, scale, label_color, 1)
            _put_text(card, str(value), x + panel_w - pad, cy, scale, value_color, 2, right_aligned=True)

        stat_row("Total Strikes", len(strikes), scale=0.85)
        cy += 42

        for punch_type in ["Jab/Cross", "Hook", "Uppercut"]:
            stat_row(f"  {punch_type}", type_counts.get(punch_type, 0), label_color=MUTED, value_color=WHITE)
            cy += 34

        cy += 16
        cv2.line(card, (cx, cy - 10), (x + panel_w - pad, cy - 10), (60, 58, 55), 1, cv2.LINE_AA)

        stat_row("Landed", result_counts.get("Landed", 0), value_color=(120, 220, 120))
        cy += 34
        stat_row("Blocked", result_counts.get("Blocked", 0), value_color=(120, 180, 240))
        cy += 34
        stat_row("Missed", result_counts.get("Missed", 0), value_color=MUTED)
        cy += 42

        cv2.line(card, (cx, cy - 10), (x + panel_w - pad, cy - 10), (60, 58, 55), 1, cv2.LINE_AA)

        rate = identity_stats.get(identity, {}).get("confirmation_rate", 0.0)
        footwork = footwork_by_identity.get(identity, {"distance_covered": 0, "dominant_stance": "unknown"})

        stat_row("Stance", footwork["dominant_stance"].capitalize(), label_color=MUTED, value_color=MUTED, scale=0.65)
        cy += 32
        stat_row("Distance Covered*", footwork["distance_covered"], label_color=MUTED, value_color=MUTED, scale=0.65)
        cy += 32
        stat_row("Glove Confirmation", f"{rate:.0%}", label_color=MUTED, value_color=MUTED, scale=0.65)

    footnote = "*Distance is a relative body-length unit, not meters -- for comparing the two fighters only."
    _put_text(card, footnote, width // 2, height - 15, 0.5, (100, 100, 100), 1, centered=True)

    return card


def save_strikes_csv(strikes_by_identity: dict, out_path: str, fps: float):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fighter", "frame", "time_seconds", "side", "punch_type",
                          "speed_score", "result", "target_zone"])
        for identity, strikes in strikes_by_identity.items():
            for s in strikes:
                writer.writerow([
                    f"Fighter {identity}", s["frame_idx"], round(s["frame_idx"] / fps, 2),
                    s["side"], s["punch_type"], s["speed"],
                    s.get("result", "Unknown"), s.get("target_zone") or "",
                ])


def save_summary_csv(strikes_by_identity: dict, identity_stats: dict, defensive_by_identity: dict,
                      footwork_by_identity: dict, out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fighter", "total_strikes", "jab_cross", "hook", "uppercut",
                          "landed", "blocked", "missed",
                          "guard_periods", "duck_slip_events",
                          "distance_covered_relative_units", "dominant_stance",
                          "frames_tracked", "glove_confirmation_rate"])
        for identity, strikes in strikes_by_identity.items():
            type_counts = defaultdict(int)
            result_counts = defaultdict(int)
            for s in strikes:
                type_counts[s["punch_type"]] += 1
                result_counts[s.get("result", "Unknown")] += 1
            stats = identity_stats.get(identity, {})
            defense = defensive_by_identity.get(identity, {"guard_periods": [], "duck_slip_events": []})
            footwork = footwork_by_identity.get(identity, {"distance_covered": 0, "dominant_stance": "unknown"})
            writer.writerow([
                f"Fighter {identity}", len(strikes),
                type_counts.get("Jab/Cross", 0), type_counts.get("Hook", 0), type_counts.get("Uppercut", 0),
                result_counts.get("Landed", 0), result_counts.get("Blocked", 0), result_counts.get("Missed", 0),
                len(defense["guard_periods"]), len(defense["duck_slip_events"]),
                footwork["distance_covered"], footwork["dominant_stance"],
                stats.get("frames_seen", 0), round(stats.get("confirmation_rate", 0.0), 3),
            ])


def run_full_pipeline(video_path, out_dir="data/output",
                       pose_model="yolov8n-pose.pt",
                       glove_model="models/glove_detector/weights/best.pt",
                       progress_callback=None,
                       show_live_panel=False):
    """
    Runs the entire pipeline on one video and returns a dict of output paths.

    progress_callback: optional function(step_number, total_steps, message)
                        called as each stage completes -- lets a UI (like
                        the Streamlit app) show live progress instead of
                        just console prints.
    """
    def report(step, total, message):
        print(message)
        if progress_callback:
            progress_callback(step, total, message)

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    TOTAL_STEPS = 6

    report(0, TOTAL_STEPS, f"=== Full pipeline: {video_path.name} ===")

    report(1, TOTAL_STEPS, "[1/6] Pose tracking + glove detection...")
    history, per_id_stats = run_glove_confirmed_tracking(
        str(video_path), pose_model_path=pose_model, glove_model_path=glove_model
    )

    report(2, TOTAL_STEPS, "[2/6] Appearance-based re-identification...")
    signatures = compute_appearance_signatures(str(video_path), history)
    track_id_to_identity, identity_stats = cluster_into_two_identities(per_id_stats, signatures)

    report(3, TOTAL_STEPS, "[3/6] Strike detection, resolution, defense, footwork...")
    frame_series_by_identity = build_identity_frame_series(history, track_id_to_identity)

    adaptive_threshold = compute_adaptive_speed_threshold(frame_series_by_identity)
    report(3, TOTAL_STEPS, f"    Auto-calculated speed threshold for this video: {adaptive_threshold:.3f}")

    strikes_by_identity = {
        identity: detect_strikes_for_identity(fs, speed_threshold=adaptive_threshold)
        for identity, fs in frame_series_by_identity.items()
    }
    strikes_by_identity = resolve_all_strikes(strikes_by_identity, history, track_id_to_identity)
    defensive_by_identity = analyze_defensive_actions(frame_series_by_identity)
    footwork_by_identity = analyze_footwork(frame_series_by_identity)

    summary_log = []
    for identity, strikes in strikes_by_identity.items():
        type_counts = defaultdict(int)
        result_counts = defaultdict(int)
        for s in strikes:
            type_counts[s["punch_type"]] += 1
            result_counts[s["result"]] += 1
        summary_log.append(f"Fighter {identity}: {len(strikes)} strikes "
                            f"({dict(type_counts)}) -- {dict(result_counts)}")
    report(4, TOTAL_STEPS, "\n".join(summary_log))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    report(5, TOTAL_STEPS, "[5/6] Rendering annotated video + stats card outro...")
    video_out_path = str(out_dir / f"{stem}_analysis.mp4")

    output_width = width + PANEL_WIDTH if show_live_panel else width
    writer = _open_video_writer(video_out_path, fps, (output_width, height))
    cap = cv2.VideoCapture(str(video_path))

    panel_state = LivePanelState(strikes_by_identity, total_frames=len(history)) if show_live_panel else None

    active_labels = defaultdict(dict)
    for identity, strikes in strikes_by_identity.items():
        for s in strikes:
            for offset in range(LABEL_DISPLAY_FRAMES):
                label_text = f"{s['punch_type']} ({s['side']}) - {s['result']}"
                active_labels[identity][s["frame_idx"] + offset] = label_text

    for identity, data in defensive_by_identity.items():
        for e in data["duck_slip_events"]:
            for offset in range(LABEL_DISPLAY_FRAMES):
                active_labels[identity][e["frame_idx"] + offset] = e["action_type"]

    guard_frame_sets = {
        identity: set(
            fi for p in data["guard_periods"] for fi in range(p["start_frame"], p["end_frame"] + 1)
        )
        for identity, data in defensive_by_identity.items()
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(history):
            for fighter in history[frame_idx]["fighters"]:
                identity = track_id_to_identity.get(fighter["track_id"])
                color = IDENTITY_COLORS.get(identity, IDENTITY_COLORS[None])
                x1, y1, x2, y2 = [int(v) for v in fighter["bbox"]]
                stance = footwork_by_identity.get(identity, {}).get("dominant_stance", "")
                label_text = f"Fighter {identity} ({stance})" if identity else "filtered"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                kpts = fighter["keypoints"]
                for a, b in SKELETON_EDGES:
                    if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
                        pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                        pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                        cv2.line(frame, pt1, pt2, color, 2)
                if identity is not None:
                    if frame_idx in guard_frame_sets.get(identity, set()):
                        cv2.putText(frame, "GUARD", (x2 - 80, max(y1 - 8, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    strike_label = active_labels[identity].get(frame_idx)
                    if strike_label:
                        cv2.putText(frame, strike_label, (x1, min(y2 + 25, height - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        if show_live_panel:
            panel_state.update(frame_idx)
            positions = {}
            if frame_idx < len(history):
                positions = estimate_positions_from_frame(history[frame_idx], track_id_to_identity, width, height)
            frame = render_frame_with_panel(frame, frame_idx, panel_state, positions, IDENTITY_COLORS)

        writer.write(frame)
        frame_idx += 1
    cap.release()

    stats_card = build_stats_card(strikes_by_identity, identity_stats, footwork_by_identity, width, height)
    if show_live_panel:
        panel_filler = np.full((height, PANEL_WIDTH, 3), (22, 20, 18), dtype=np.uint8)
        stats_card = np.hstack([stats_card, panel_filler])
    for _ in range(int(STATS_CARD_SECONDS * fps)):
        writer.write(stats_card)
    writer.release()

    report(6, TOTAL_STEPS, "[6/6] Saving CSV stats...")
    strikes_csv_path = str(out_dir / f"{stem}_strikes.csv")
    summary_csv_path = str(out_dir / f"{stem}_summary.csv")
    summary_png_path = str(out_dir / f"{stem}_summary.png")

    save_strikes_csv(strikes_by_identity, strikes_csv_path, fps)
    save_summary_csv(strikes_by_identity, identity_stats, defensive_by_identity,
                      footwork_by_identity, summary_csv_path)
    cv2.imwrite(summary_png_path, stats_card)

    return {
        "video": video_out_path,
        "strikes_csv": strikes_csv_path,
        "summary_csv": summary_csv_path,
        "summary_png": summary_png_path,
        "strikes_by_identity": strikes_by_identity,
        "identity_stats": identity_stats,
        "defensive_by_identity": defensive_by_identity,
        "footwork_by_identity": footwork_by_identity,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full pipeline: video in, full analytics out")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", default="data/output")
    parser.add_argument("--pose_model", default="yolov8n-pose.pt")
    parser.add_argument("--glove_model", default="models/glove_detector/weights/best.pt")
    parser.add_argument("--live_panel", action="store_true",
                         help="Add a broadcast-style side panel (hit zones, position map, activity graph)")
    args = parser.parse_args()

    result = run_full_pipeline(args.video, args.out_dir, args.pose_model, args.glove_model,
                                show_live_panel=args.live_panel)

    print(f"\nDone. Outputs in {args.out_dir}/:")
    print(f"  {Path(result['video']).name}")
    print(f"  {Path(result['strikes_csv']).name}")
    print(f"  {Path(result['summary_csv']).name}")
    print(f"  {Path(result['summary_png']).name}")
