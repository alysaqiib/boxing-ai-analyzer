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


def _round_number(frame_idx: int, fps: float, round_duration_seconds: float) -> int:
    """Return a one-based round number for a frame using fixed time windows."""
    return int(frame_idx / (fps * round_duration_seconds)) + 1


def save_strikes_csv(strikes_by_identity: dict, out_path: str, fps: float,
                     round_duration_seconds: float):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "fighter", "frame", "time_seconds", "side", "punch_type",
                          "speed_score", "confidence_percent", "result", "target_zone"])
        for identity, strikes in strikes_by_identity.items():
            for s in strikes:
                writer.writerow([
                    _round_number(s["frame_idx"], fps, round_duration_seconds),
                    f"Fighter {identity}", s["frame_idx"], round(s["frame_idx"] / fps, 2),
                    s["side"], s["punch_type"], s["speed"], s.get("confidence_percent", 0),
                    s.get("result", "Unknown"), s.get("target_zone") or "",
                ])


def add_strike_confidence_scores(strikes_by_identity: dict, speed_threshold: float) -> dict:
    """Add a transparent 0-100 confidence estimate to each detected strike."""
    if speed_threshold <= 0:
        raise ValueError("speed_threshold must be greater than zero")

    scored = {}
    for identity, strikes in strikes_by_identity.items():
        scored[identity] = []
        for strike in strikes:
            speed_ratio = min(float(strike.get("speed", 0.0)) / speed_threshold, 2.0)
            motion_score = min(speed_ratio / 2.0, 1.0)
            result_score = 0.65 if strike.get("result") == "Unknown" else 1.0
            confidence = round(100 * motion_score * result_score)
            enriched = dict(strike)
            enriched["confidence_percent"] = confidence
            scored[identity].append(enriched)
    return scored


def filter_unresolved_strikes(strikes_by_identity: dict) -> dict:
    """Keep only strikes whose outcome could be resolved against the opponent."""
    return {
        identity: [
            strike for strike in strikes
            if strike.get("result") in {"Landed", "Blocked", "Missed"}
        ]
        for identity, strikes in strikes_by_identity.items()
    }


def build_round_summary(strikes_by_identity: dict, defensive_by_identity: dict,
                        footwork_by_identity: dict, frame_series_by_identity: dict,
                        fps: float, total_frames: int,
                        round_duration_seconds: float) -> list:
    """Build one comparable analytics row per fighter and time-based round."""
    if round_duration_seconds <= 0:
        raise ValueError("round_duration_seconds must be greater than zero")

    total_rounds = max(1, _round_number(max(total_frames - 1, 0), fps, round_duration_seconds))
    rows = []
    for round_number in range(1, total_rounds + 1):
        start_frame = int((round_number - 1) * fps * round_duration_seconds)
        end_frame = min(int(round_number * fps * round_duration_seconds) - 1, max(total_frames - 1, 0))
        for identity in sorted(set(strikes_by_identity) | set(frame_series_by_identity)):
            strikes = [
                s for s in strikes_by_identity.get(identity, [])
                if start_frame <= s["frame_idx"] <= end_frame
            ]
            type_counts = defaultdict(int)
            result_counts = defaultdict(int)
            zone_counts = defaultdict(int)
            for strike in strikes:
                type_counts[strike["punch_type"]] += 1
                result_counts[strike.get("result", "Unknown")] += 1
                if strike.get("target_zone"):
                    zone_counts[strike["target_zone"]] += 1

            defense = defensive_by_identity.get(identity, {})
            guard_periods = [
                period for period in defense.get("guard_periods", [])
                if period["end_frame"] >= start_frame and period["start_frame"] <= end_frame
            ]
            duck_slip_events = [
                event for event in defense.get("duck_slip_events", [])
                if start_frame <= event["frame_idx"] <= end_frame
            ]
            frame_series = {
                frame: data for frame, data in frame_series_by_identity.get(identity, {}).items()
                if start_frame <= frame <= end_frame
            }
            footwork = analyze_footwork({identity: frame_series}).get(
                identity, {"distance_covered": 0, "dominant_stance": "unknown"}
            )
            total_strikes = len(strikes)
            landed = result_counts.get("Landed", 0)
            rows.append({
                "round": round_number,
                "fighter": f"Fighter {identity}",
                "start_time_seconds": round(start_frame / fps, 2),
                "end_time_seconds": round((end_frame + 1) / fps, 2),
                "total_strikes": total_strikes,
                "jab_cross": type_counts.get("Jab/Cross", 0),
                "hook": type_counts.get("Hook", 0),
                "uppercut": type_counts.get("Uppercut", 0),
                "landed": landed,
                "blocked": result_counts.get("Blocked", 0),
                "missed": result_counts.get("Missed", 0),
                "unknown": result_counts.get("Unknown", 0),
                "accuracy_percent": round(100 * landed / total_strikes, 1) if total_strikes else 0.0,
                "head_targets": zone_counts.get("Head", 0),
                "body_targets": zone_counts.get("Body", 0),
                "leg_targets": zone_counts.get("Leg", 0),
                "guard_periods": len(guard_periods),
                "duck_slip_events": len(duck_slip_events),
                "distance_covered_relative_units": footwork["distance_covered"],
                "dominant_stance": footwork["dominant_stance"],
                "frames_tracked": len(frame_series),
            })
    return rows


def save_round_summary_csv(round_summary: list, out_path: str):
    """Save round-by-round analytics in a spreadsheet-friendly format."""
    fields = [
        "round", "fighter", "start_time_seconds", "end_time_seconds",
        "total_strikes", "jab_cross", "hook", "uppercut", "landed", "blocked",
        "missed", "unknown", "accuracy_percent", "head_targets", "body_targets",
        "leg_targets", "guard_periods", "duck_slip_events",
        "distance_covered_relative_units", "dominant_stance", "frames_tracked",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(round_summary)


def build_combinations(strikes_by_identity: dict, fps: float,
                       max_gap_seconds: float = 0.75) -> list:
    """Group nearby strikes from each fighter into two-or-more-punch combinations."""
    if fps <= 0 or max_gap_seconds <= 0:
        raise ValueError("fps and max_gap_seconds must be greater than zero")

    max_gap_frames = max(1, int(fps * max_gap_seconds))
    combinations = []
    for identity, strikes in strikes_by_identity.items():
        ordered = sorted(strikes, key=lambda strike: strike["frame_idx"])
        current = []
        for strike in ordered:
            if current and strike["frame_idx"] - current[-1]["frame_idx"] > max_gap_frames:
                if len(current) >= 2:
                    combinations.append(_format_combination(identity, current, fps))
                current = []
            current.append(strike)
        if len(current) >= 2:
            combinations.append(_format_combination(identity, current, fps))
    return sorted(combinations, key=lambda combo: combo["start_frame"])


def _format_combination(identity: str, strikes: list, fps: float) -> dict:
    result_counts = defaultdict(int)
    for strike in strikes:
        result_counts[strike.get("result", "Unknown")] += 1
    return {
        "fighter": f"Fighter {identity}",
        "start_frame": strikes[0]["frame_idx"],
        "end_frame": strikes[-1]["frame_idx"],
        "start_time_seconds": round(strikes[0]["frame_idx"] / fps, 2),
        "end_time_seconds": round(strikes[-1]["frame_idx"] / fps, 2),
        "punch_count": len(strikes),
        "sequence": " -> ".join(strike["punch_type"] for strike in strikes),
        "landed": result_counts.get("Landed", 0),
        "blocked": result_counts.get("Blocked", 0),
        "missed": result_counts.get("Missed", 0),
        "unknown": result_counts.get("Unknown", 0),
    }


def save_combinations_csv(combinations: list, out_path: str):
    fields = [
        "fighter", "start_frame", "end_frame", "start_time_seconds",
        "end_time_seconds", "punch_count", "sequence", "landed", "blocked",
        "missed", "unknown",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combinations)


def save_summary_csv(strikes_by_identity: dict, identity_stats: dict, defensive_by_identity: dict,
                      footwork_by_identity: dict, out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fighter", "total_strikes", "jab_cross", "hook", "uppercut",
                          "landed", "blocked", "missed",
                          "accuracy_percent", "head_targets", "body_targets", "leg_targets",
                          "guard_periods", "duck_slip_events",
                          "distance_covered_relative_units", "dominant_stance",
                          "frames_tracked", "glove_confirmation_rate"])
        for identity, strikes in strikes_by_identity.items():
            type_counts = defaultdict(int)
            result_counts = defaultdict(int)
            for s in strikes:
                type_counts[s["punch_type"]] += 1
                result_counts[s.get("result", "Unknown")] += 1
            target_counts = defaultdict(int)
            for strike in strikes:
                if strike.get("target_zone"):
                    target_counts[strike["target_zone"]] += 1
            stats = identity_stats.get(identity, {})
            defense = defensive_by_identity.get(identity, {"guard_periods": [], "duck_slip_events": []})
            footwork = footwork_by_identity.get(identity, {"distance_covered": 0, "dominant_stance": "unknown"})
            landed = result_counts.get("Landed", 0)
            writer.writerow([
                f"Fighter {identity}", len(strikes),
                type_counts.get("Jab/Cross", 0), type_counts.get("Hook", 0), type_counts.get("Uppercut", 0),
                landed, result_counts.get("Blocked", 0), result_counts.get("Missed", 0),
                round(100 * landed / len(strikes), 1) if strikes else 0.0,
                target_counts.get("Head", 0), target_counts.get("Body", 0), target_counts.get("Leg", 0),
                len(defense["guard_periods"]), len(defense["duck_slip_events"]),
                footwork["distance_covered"], footwork["dominant_stance"],
                stats.get("frames_seen", 0), round(stats.get("confirmation_rate", 0.0), 3),
            ])


def calculate_performance_scores(strikes_by_identity: dict, defensive_by_identity: dict,
                                 footwork_by_identity: dict, identity_stats: dict) -> list:
    """Return explainable 0-100 scores and coaching feedback per fighter."""
    totals = {
        identity: len(strikes)
        for identity, strikes in strikes_by_identity.items()
    }
    max_total = max(max(totals.values(), default=0), 1)
    rows = []
    for identity in sorted(set(strikes_by_identity) | set(identity_stats)):
        strikes = strikes_by_identity.get(identity, [])
        result_counts = defaultdict(int)
        for strike in strikes:
            result_counts[strike.get("result", "Unknown")] += 1
        total = len(strikes)
        landed = result_counts["Landed"]
        accuracy = 100 * landed / total if total else 0.0
        activity = 100 * totals.get(identity, 0) / max_total
        defense = defensive_by_identity.get(identity, {})
        defensive_events = len(defense.get("duck_slip_events", []))
        guard_periods = len(defense.get("guard_periods", []))
        defensive_activity = min(100.0, defensive_events * 10.0 + guard_periods * 15.0)
        footwork = footwork_by_identity.get(identity, {})
        movement = min(100.0, float(footwork.get("distance_covered", 0.0)) * 10.0)
        confirmation = 100 * float(identity_stats.get(identity, {}).get("confirmation_rate", 0.0))
        score = round(
            accuracy * 0.30
            + activity * 0.25
            + defensive_activity * 0.20
            + movement * 0.15
            + confirmation * 0.10
        )
        feedback = []
        if accuracy < 35:
            feedback.append("Prioritize accuracy over throwing volume.")
        elif accuracy >= 60:
            feedback.append("Maintain the current punch accuracy.")
        if activity < 50:
            feedback.append("Increase controlled punch activity.")
        if defensive_activity < 30:
            feedback.append("Work on guard, slips, and ducks after exchanges.")
        if movement < 35:
            feedback.append("Add more purposeful footwork and exits.")
        if not feedback:
            feedback.append("Balanced performance across the measured categories.")
        rows.append({
            "fighter": f"Fighter {identity}",
            "overall_score": score,
            "accuracy_score": round(accuracy, 1),
            "activity_score": round(activity, 1),
            "defensive_activity_score": round(defensive_activity, 1),
            "movement_score": round(movement, 1),
            "tracking_confidence_score": round(confirmation, 1),
            "coaching_feedback": " ".join(feedback),
        })
    return rows


def save_performance_scores_csv(scores: list, out_path: str):
    fields = [
        "fighter", "overall_score", "accuracy_score", "activity_score",
        "defensive_activity_score", "movement_score", "tracking_confidence_score",
        "coaching_feedback",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scores)


def rebuild_corrected_reports(corrected_rows: list, result: dict, out_dir: str,
                              stem: str, round_duration_seconds: float) -> dict:
    """Rebuild derived reports after manual strike-field corrections.

    The detector output and annotated video remain untouched. Only fields
    supplied by the editable strike table are used to recalculate reports.
    """
    corrected_by_identity = defaultdict(list)
    for row in corrected_rows:
        fighter = str(row.get("fighter", ""))
        identity = fighter.replace("Fighter ", "", 1).strip()
        if identity not in {"A", "B"}:
            raise ValueError(f"Invalid fighter value in corrected strike data: {fighter!r}")
        try:
            frame_idx = int(row["frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Corrected strike data contains an invalid frame.") from error
        target_zone = str(row.get("target_zone", "") or "").strip() or None
        corrected_by_identity[identity].append({
            "frame_idx": frame_idx,
            "side": str(row.get("side", "")),
            "punch_type": str(row.get("punch_type", "Jab/Cross")),
            "speed": float(row.get("speed_score", 0.0)),
            "confidence_percent": float(row.get("confidence_percent", 0.0)),
            "result": str(row.get("result", "Unknown")),
            "target_zone": target_zone,
        })

    out_path = Path(out_dir)
    corrected_strikes_csv = str(out_path / f"{stem}_strikes_corrected.csv")
    corrected_summary_csv = str(out_path / f"{stem}_summary_corrected.csv")
    corrected_round_csv = str(out_path / f"{stem}_round_summary_corrected.csv")
    corrected_combinations_csv = str(out_path / f"{stem}_combinations_corrected.csv")
    corrected_performance_csv = str(out_path / f"{stem}_performance_corrected.csv")
    corrected_summary_png = str(out_path / f"{stem}_summary_corrected.png")

    fps = float(result["fps"])
    save_strikes_csv(
        corrected_by_identity, corrected_strikes_csv, fps, round_duration_seconds
    )
    save_summary_csv(
        corrected_by_identity, result["identity_stats"],
        result["defensive_by_identity"], result["footwork_by_identity"],
        corrected_summary_csv,
    )
    round_summary = build_round_summary(
        corrected_by_identity, result["defensive_by_identity"],
        result["footwork_by_identity"], result["frame_series_by_identity"],
        fps, result["total_frames"], round_duration_seconds,
    )
    save_round_summary_csv(round_summary, corrected_round_csv)
    combinations = build_combinations(corrected_by_identity, fps)
    save_combinations_csv(combinations, corrected_combinations_csv)
    performance_scores = calculate_performance_scores(
        corrected_by_identity, result["defensive_by_identity"],
        result["footwork_by_identity"], result["identity_stats"],
    )
    save_performance_scores_csv(performance_scores, corrected_performance_csv)
    summary_card = build_stats_card(
        corrected_by_identity, result["identity_stats"],
        result["footwork_by_identity"], result["video_width"], result["video_height"],
    )
    cv2.imwrite(corrected_summary_png, summary_card)

    return {
        "strikes_csv": corrected_strikes_csv,
        "summary_csv": corrected_summary_csv,
        "summary_png": corrected_summary_png,
        "round_summary_csv": corrected_round_csv,
        "round_summary": round_summary,
        "combinations_csv": corrected_combinations_csv,
        "combinations": combinations,
        "performance_csv": corrected_performance_csv,
        "performance_scores": performance_scores,
        "strikes_by_identity": dict(corrected_by_identity),
    }


def run_full_pipeline(video_path, out_dir="data/output",
                       pose_model="yolov8n-pose.pt",
                       glove_model="models/glove_detector/weights/best.pt",
                       progress_callback=None,
                       show_live_panel=False,
                       round_duration_seconds=180.0):
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

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    report(3, TOTAL_STEPS, "[3/6] Strike detection, resolution, defense, footwork...")
    frame_series_by_identity = build_identity_frame_series(history, track_id_to_identity)

    adaptive_threshold = compute_adaptive_speed_threshold(frame_series_by_identity, fps=fps)
    report(3, TOTAL_STEPS, f"    Auto-calculated speed threshold: {adaptive_threshold:.3f}/s at {fps:.2f} FPS")

    strikes_by_identity = {
        identity: detect_strikes_for_identity(fs, speed_threshold=adaptive_threshold, fps=fps)
        for identity, fs in frame_series_by_identity.items()
    }
    strikes_by_identity = resolve_all_strikes(strikes_by_identity, history, track_id_to_identity)
    strikes_by_identity = add_strike_confidence_scores(strikes_by_identity, adaptive_threshold)
    strikes_by_identity = filter_unresolved_strikes(strikes_by_identity)
    defensive_by_identity = analyze_defensive_actions(frame_series_by_identity)
    footwork_by_identity = analyze_footwork(frame_series_by_identity)
    if round_duration_seconds <= 0:
        raise ValueError("round_duration_seconds must be greater than zero")

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

    save_strikes_csv(strikes_by_identity, strikes_csv_path, fps, round_duration_seconds)
    save_summary_csv(strikes_by_identity, identity_stats, defensive_by_identity,
                      footwork_by_identity, summary_csv_path)
    round_summary = build_round_summary(
        strikes_by_identity, defensive_by_identity, footwork_by_identity,
        frame_series_by_identity, fps, len(history), round_duration_seconds,
    )
    round_summary_csv_path = str(out_dir / f"{stem}_round_summary.csv")
    save_round_summary_csv(round_summary, round_summary_csv_path)
    combinations = build_combinations(strikes_by_identity, fps)
    combinations_csv_path = str(out_dir / f"{stem}_combinations.csv")
    save_combinations_csv(combinations, combinations_csv_path)
    performance_scores = calculate_performance_scores(
        strikes_by_identity, defensive_by_identity, footwork_by_identity, identity_stats
    )
    performance_csv_path = str(out_dir / f"{stem}_performance.csv")
    save_performance_scores_csv(performance_scores, performance_csv_path)
    cv2.imwrite(summary_png_path, stats_card)

    return {
        "video": video_out_path,
        "strikes_csv": strikes_csv_path,
        "summary_csv": summary_csv_path,
        "summary_png": summary_png_path,
        "round_summary_csv": round_summary_csv_path,
        "round_summary": round_summary,
        "combinations_csv": combinations_csv_path,
        "combinations": combinations,
        "performance_csv": performance_csv_path,
        "performance_scores": performance_scores,
        "fps": fps,
        "strikes_by_identity": strikes_by_identity,
        "identity_stats": identity_stats,
        "defensive_by_identity": defensive_by_identity,
        "footwork_by_identity": footwork_by_identity,
        "frame_series_by_identity": frame_series_by_identity,
        "total_frames": len(history),
        "video_width": width,
        "video_height": height,
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
    parser.add_argument("--round_duration", type=float, default=180.0,
                        help="Length of each analytics round in seconds (default: 180)")
    args = parser.parse_args()

    result = run_full_pipeline(args.video, args.out_dir, args.pose_model, args.glove_model,
                                show_live_panel=args.live_panel,
                                round_duration_seconds=args.round_duration)

    print(f"\nDone. Outputs in {args.out_dir}/:")
    print(f"  {Path(result['video']).name}")
    print(f"  {Path(result['strikes_csv']).name}")
    print(f"  {Path(result['summary_csv']).name}")
    print(f"  {Path(result['round_summary_csv']).name}")
    print(f"  {Path(result['combinations_csv']).name}")
    print(f"  {Path(result['performance_csv']).name}")
    print(f"  {Path(result['summary_png']).name}")
