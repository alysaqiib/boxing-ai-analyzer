"""
Phase 3: Strike classification
----------------------------------
Classifies punches (jab/cross, hook, uppercut) using wrist velocity
and joint angles from the pose data we already have -- no new model
or dataset needed.

HOW IT WORKS
For each identified fighter (from Phase 2c), we track wrist speed
frame-to-frame. A "strike" is a sharp spike in wrist speed. At the
peak of that spike, we classify the punch type using:
  - Elbow angle (shoulder-elbow-wrist) -- extended arm vs bent arm
  - Vertical vs horizontal wrist displacement during the strike

HONEST LIMITATIONS (please read before trusting the output)
  - This is a heuristic, not a trained classifier -- expect real
    misclassifications, especially jab-vs-hook confusion. From a
    single 2D side-camera angle, a jab (straight, toward camera)
    and a hook (lateral) can look very similar without true depth
    information. Treat this as a first draft to review and tune
    against your own footage, not ground truth.
  - Feints, blocks, and fast defensive hand movement can trigger
    false "strikes". There's no separate defense-detection yet.
  - Thresholds (SPEED_THRESHOLD, MIN_GAP_FRAMES, etc.) were picked
    to be reasonable defaults -- you WILL need to tune them by
    watching your own output and adjusting.
"""

import cv2
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

from pose_tracker import SKELETON_EDGES
from reid_fighter_tracker import (
    run_glove_confirmed_tracking,
    compute_appearance_signatures,
    cluster_into_two_identities,
    IDENTITY_COLORS,
)

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10

KEYPOINT_CONF_THRESHOLD = 0.3

# Tune these against your own footage -- defaults are a starting point.
SPEED_THRESHOLD = 0.35       # fallback normalized wrist speed per second
MIN_ADAPTIVE_SPEED = 0.12    # safety floor for unusual/very short clips
MAX_ADAPTIVE_SPEED = 10.0    # broad safety ceiling; threshold remains video-dependent
MIN_ARM_EXTENSION_GAIN = 0.10  # wrist-to-shoulder distance increase, normalized by bbox diagonal
MIN_WRIST_SHOULDER_DISTANCE = 0.20  # minimum extended-arm distance at the candidate peak
MIN_ELBOW_MOTION = 0.04      # require the elbow to move with the wrist, rejecting keypoint jumps
MIN_RELATIVE_ARM_MOTION = 0.12  # wrist must move relative to the shoulder, not with the whole body
MIN_GAP_FRAMES = 5           # minimum frames between two separate strikes (same wrist)
MIN_CROSS_WRIST_GAP_FRAMES = 3  # two hands cannot produce separate punches in adjacent frames
STRIKE_WINDOW = 4            # frames to look back from the peak, for displacement/angle measurement
LABEL_DISPLAY_FRAMES = 8     # how many frames to keep a strike label visible after detection


def _joint_angle(a, b, c):
    """Angle at point b, formed by points a-b-c, in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _bbox_diag(bbox):
    x1, y1, x2, y2 = bbox
    return float(np.hypot(x2 - x1, y2 - y1)) + 1e-6


def build_identity_frame_series(history: list, track_id_to_identity: dict) -> dict:
    """Returns identity -> {frame_idx: fighter_entry} for easy lookups."""
    series = defaultdict(dict)
    for frame_data in history:
        for fighter in frame_data["fighters"]:
            identity = track_id_to_identity.get(fighter["track_id"])
            if identity is not None:
                series[identity][frame_data["frame_idx"]] = fighter
    return series


def _wrist_speed_series(frame_series: dict, wrist_idx: int, fps: float = 30.0) -> list:
    """Return normalized wrist speed per second, independent of source FPS."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    frame_indices = sorted(frame_series.keys())
    speeds = []
    prev_frame_idx, prev_pos = None, None

    for fi in frame_indices:
        kpts = frame_series[fi]["keypoints"]
        wrist = kpts[wrist_idx]
        if wrist[2] < KEYPOINT_CONF_THRESHOLD:
            prev_frame_idx, prev_pos = None, None
            continue

        pos = np.array([wrist[0], wrist[1]])
        if prev_pos is not None and prev_frame_idx is not None:
            gap = fi - prev_frame_idx
            if gap > 0:
                dist = np.linalg.norm(pos - prev_pos)
                diag = _bbox_diag(frame_series[fi]["bbox"])
                speeds.append((fi, (dist / diag) * fps / gap))
        prev_frame_idx, prev_pos = fi, pos

    return speeds


def compute_adaptive_speed_threshold(frame_series_by_identity: dict, percentile: float = 92,
                                      min_threshold: float = MIN_ADAPTIVE_SPEED,
                                      max_threshold: float = MAX_ADAPTIVE_SPEED,
                                      fps: float = 30.0) -> float:
    """
    Automatically picks a SPEED_THRESHOLD based on THIS video's own movement data,
    instead of using one fixed number for every video. Different videos have very
    different typical motion scales (camera distance, punching style, sparring vs.
    real match pace) -- a threshold tuned on one video often doesn't transfer to another.

    Pools wrist-speed data across both fighters, takes the given percentile (so only
    the fastest ~8% of movements by default count as "strike-like"), and clips the
    result to a sane range so it never ends up absurdly high or low on unusual footage.
    """
    all_speeds = []
    for identity, frame_series in frame_series_by_identity.items():
        for wrist_idx in [LEFT_WRIST, RIGHT_WRIST]:
            all_speeds.extend(
                speed for _, speed in _wrist_speed_series(frame_series, wrist_idx, fps=fps)
            )

    if not all_speeds:
        return SPEED_THRESHOLD  # fallback to the default constant if no data at all

    threshold = float(np.percentile(all_speeds, percentile))
    return float(np.clip(threshold, max(min_threshold, MIN_ADAPTIVE_SPEED), max_threshold))


def detect_strikes_for_identity(frame_series: dict, speed_threshold: float = None,
                                fps: float = 30.0) -> list:
    """
    frame_series: {frame_idx: fighter_entry} for ONE identity, sorted implicitly by frame_idx.
    speed_threshold: if None, falls back to the module-level SPEED_THRESHOLD constant
                      (useful for quick standalone testing). Normally you should pass in
                      a value from compute_adaptive_speed_threshold() instead.
    Returns a list of detected strikes: {frame_idx, side, punch_type, speed}.
    """
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if speed_threshold is None:
        speed_threshold = SPEED_THRESHOLD

    frame_indices = sorted(frame_series.keys())
    strikes = []

    for side_name, wrist_idx, elbow_idx, shoulder_idx in [
        ("left", LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER),
        ("right", RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER),
    ]:
        speeds = _wrist_speed_series(frame_series, wrist_idx, fps=fps)
        positions = {}
        for fi in frame_indices:
            kpts = frame_series[fi]["keypoints"]
            wrist = kpts[wrist_idx]
            if wrist[2] >= KEYPOINT_CONF_THRESHOLD:
                positions[fi] = np.array([wrist[0], wrist[1]])

        # Find local peaks above threshold, spaced apart
        last_strike_frame = -MIN_GAP_FRAMES
        for i in range(1, len(speeds) - 1):
            fi, spd = speeds[i]
            prev_spd = speeds[i - 1][1]
            next_spd = speeds[i + 1][1]

            is_local_peak = spd >= prev_spd and spd >= next_spd
            if (
                is_local_peak
                and spd >= float(speed_threshold)
                and (fi - last_strike_frame) >= MIN_GAP_FRAMES
                and all(
                    abs(fi - previous["frame_idx"]) >= MIN_CROSS_WRIST_GAP_FRAMES
                    for previous in strikes
                )
            ):
                # Classify using displacement + elbow angle
                start_fi = max(fi - STRIKE_WINDOW, frame_indices[0])
                if start_fi not in positions or fi not in positions:
                    continue

                start_pos = positions.get(start_fi, positions[fi])
                peak_pos = positions[fi]

                vertical_disp = start_pos[1] - peak_pos[1]     # positive = moved up (image y grows downward)
                horizontal_disp = peak_pos[0] - start_pos[0]

                start_kpts = frame_series[start_fi]["keypoints"]
                kpts = frame_series[fi]["keypoints"]
                required_indices = [shoulder_idx, elbow_idx, wrist_idx]
                if any(kpts[index][2] < KEYPOINT_CONF_THRESHOLD for index in required_indices):
                    continue
                bbox_diag = _bbox_diag(frame_series[fi]["bbox"])
                start_arm_length = np.linalg.norm(start_pos - start_kpts[shoulder_idx][:2]) / bbox_diag
                peak_arm_length = np.linalg.norm(peak_pos - kpts[shoulder_idx][:2]) / bbox_diag
                elbow_motion = (
                    np.linalg.norm(kpts[elbow_idx][:2] - start_kpts[elbow_idx][:2])
                    / bbox_diag
                )
                relative_arm_motion = np.linalg.norm(
                    (peak_pos - start_pos)
                    - (kpts[shoulder_idx][:2] - start_kpts[shoulder_idx][:2])
                ) / bbox_diag
                if (
                    peak_arm_length < MIN_WRIST_SHOULDER_DISTANCE
                    or peak_arm_length - start_arm_length < MIN_ARM_EXTENSION_GAIN
                    or elbow_motion < MIN_ELBOW_MOTION
                    or relative_arm_motion < MIN_RELATIVE_ARM_MOTION
                ):
                    continue
                shoulder = kpts[shoulder_idx][:2]
                elbow = kpts[elbow_idx][:2]
                wrist = kpts[wrist_idx][:2]

                # Check the peak frame AND a couple frames after it, use whichever
                # is straightest -- full extension often lands just after the
                # wrist-speed peak, not exactly at it.
                best_elbow_angle = _joint_angle(shoulder, elbow, wrist)
                for check_fi in range(fi, min(fi + 3, frame_indices[-1] + 1)):
                    if check_fi in positions:
                        k2 = frame_series[check_fi]["keypoints"]
                        angle2 = _joint_angle(k2[shoulder_idx][:2], k2[elbow_idx][:2], k2[wrist_idx][:2])
                        if angle2 > best_elbow_angle:
                            best_elbow_angle = angle2
                elbow_angle = best_elbow_angle

                if abs(vertical_disp) > abs(horizontal_disp) * 1.2 and vertical_disp > 0 and elbow_angle < 130:
                    punch_type = "Uppercut"
                elif elbow_angle > 130:
                    punch_type = "Jab/Cross"
                else:
                    punch_type = "Hook"

                strikes.append({
                    "frame_idx": fi,
                    "side": side_name,
                    "punch_type": punch_type,
                    "speed": round(float(spd), 3),
                })
                last_strike_frame = fi

    strikes.sort(key=lambda s: s["frame_idx"])
    return strikes


def render_strike_video(video_path: str, history: list, track_id_to_identity: dict,
                         strikes_by_identity: dict, out_path: str):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _open_writer(path, fps, size):
        for code in ["avc1", "H264", "mp4v"]:
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
            if w.isOpened():
                return w
            w.release()
        raise RuntimeError(f"Could not open video writer for {path}")

    writer = _open_writer(out_path, fps, (width, height))

    # Build a quick lookup: for each identity, frame_idx -> punch label (active for a few frames after detection)
    active_labels = defaultdict(dict)  # identity -> {frame_idx: label_text}
    for identity, strikes in strikes_by_identity.items():
        for s in strikes:
            for offset in range(LABEL_DISPLAY_FRAMES):
                active_labels[identity][s["frame_idx"] + offset] = f"{s['punch_type']} ({s['side']})"

    counts = defaultdict(int)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(history):
            for fighter in history[frame_idx]["fighters"]:
                identity = track_id_to_identity.get(fighter["track_id"])
                color = IDENTITY_COLORS.get(identity, IDENTITY_COLORS[None])
                if identity is None:
                    continue  # don't draw filtered/non-fighters for this view

                x1, y1, x2, y2 = [int(v) for v in fighter["bbox"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Fighter {identity}", (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                kpts = fighter["keypoints"]
                for a, b in SKELETON_EDGES:
                    if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
                        pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                        pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                        cv2.line(frame, pt1, pt2, color, 2)

                label = active_labels[identity].get(frame_idx)
                if label:
                    cv2.putText(frame, label, (x1, min(y2 + 25, height - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def save_strikes_csv(strikes_by_identity: dict, out_path: str, fps: float):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fighter", "frame", "time_seconds", "side", "punch_type", "speed_score"])
        for identity, strikes in strikes_by_identity.items():
            for s in strikes:
                writer.writerow([
                    f"Fighter {identity}",
                    s["frame_idx"],
                    round(s["frame_idx"] / fps, 2),
                    s["side"],
                    s["punch_type"],
                    s["speed"],
                ])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: strike classification")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="data/output/strikes.mp4")
    parser.add_argument("--csv", default="data/output/strikes.csv")
    parser.add_argument("--pose_model", default="yolov8n-pose.pt")
    parser.add_argument("--glove_model", default="models/glove_detector/weights/best.pt")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("Step 1/4: running pose tracking + glove detection...")
    history, per_id_stats = run_glove_confirmed_tracking(
        args.video, pose_model_path=args.pose_model, glove_model_path=args.glove_model
    )

    print("Step 2/4: computing appearance signatures and identities...")
    signatures = compute_appearance_signatures(args.video, history)
    track_id_to_identity, identity_stats = cluster_into_two_identities(per_id_stats, signatures)

    print("Step 3/4: detecting and classifying strikes...")
    frame_series_by_identity = build_identity_frame_series(history, track_id_to_identity)

    adaptive_threshold = compute_adaptive_speed_threshold(frame_series_by_identity)
    print(f"\nAuto-calculated SPEED_THRESHOLD for this video: {adaptive_threshold:.3f} "
          f"(based on the 92nd percentile of this video's own wrist-speed data)")

    strikes_by_identity = {}
    for identity, frame_series in frame_series_by_identity.items():
        strikes_by_identity[identity] = detect_strikes_for_identity(frame_series, speed_threshold=adaptive_threshold)

    print("\n--- Diagnostic: speed distribution per identity (informational) ---")
    for identity, frame_series in frame_series_by_identity.items():
        frame_indices = sorted(frame_series.keys())
        for side_name, wrist_idx in [("left", LEFT_WRIST), ("right", RIGHT_WRIST)]:
            all_speeds = []
            prev_fi, prev_pos = None, None
            for fi in frame_indices:
                kpts = frame_series[fi]["keypoints"]
                wrist = kpts[wrist_idx]
                if wrist[2] < KEYPOINT_CONF_THRESHOLD:
                    prev_fi, prev_pos = None, None
                    continue
                pos = np.array([wrist[0], wrist[1]])
                if prev_pos is not None:
                    gap = fi - prev_fi
                    if gap > 0:
                        dist = np.linalg.norm(pos - prev_pos)
                        diag = _bbox_diag(frame_series[fi]["bbox"])
                        all_speeds.append((dist / diag) / gap)
                prev_fi, prev_pos = fi, pos
            if all_speeds:
                arr = np.array(all_speeds)
                print(f"  Fighter {identity} {side_name} wrist: max={arr.max():.3f}, "
                      f"mean={arr.mean():.3f}, 95th_pct={np.percentile(arr, 95):.3f}")

    for identity, strikes in strikes_by_identity.items():
        print(f"\nFighter {identity}: {len(strikes)} strikes detected")
        type_counts = defaultdict(int)
        for s in strikes:
            type_counts[s["punch_type"]] += 1
        for punch_type, count in type_counts.items():
            print(f"    {punch_type}: {count}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    print("\nStep 4/4: rendering output video and CSV...")
    render_strike_video(args.video, history, track_id_to_identity, strikes_by_identity, args.out)
    save_strikes_csv(strikes_by_identity, args.csv, fps)

    print(f"\nDone. Video saved to: {args.out}")
    print(f"CSV saved to: {args.csv}")
    print("\nIMPORTANT: this is a heuristic classifier, not a trained model. Watch the output "
          "video and compare against what you know actually happened -- expect to need to "
          "tune SPEED_THRESHOLD and the classification rules at the top of this file.")
