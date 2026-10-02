"""
Phase 5: Defensive action detection
----------------------------------------
Detects three defensive behaviors using pose keypoints we already have:

  GUARD  -- both wrists held close to the head for a sustained stretch
            (a static defensive posture, not a punch-like spike)
  DUCK   -- head drops sharply relative to the torso, then usually
            recovers (bending knees/waist to go under a punch)
  SLIP   -- head shifts sideways relative to the hips without much
            vertical drop (weaving to the side)

HONEST LIMITATIONS
  - Heuristic, not a trained classifier -- same caveat as strike
    classification. Expect to tune thresholds against your own footage.
  - A duck/slip and an actual strike can look similar in raw motion
    terms; we don't yet cross-check against strike_classifier.py to
    rule out double-counting (e.g. a big head bob from taking a hit
    could register as a "duck").
  - Guard detection only checks wrist-to-head distance, not whether
    the guard is actually being used defensively vs. just resting.
"""

import numpy as np
from collections import defaultdict

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_WRIST, RIGHT_WRIST = 9, 10

KEYPOINT_CONF_THRESHOLD = 0.3

# Tune these against real footage, same as strike_classifier's thresholds.
GUARD_DISTANCE_THRESHOLD = 0.45   # wrist-to-nose distance (normalized by torso height) to count as "guarding"
GUARD_MIN_CONSECUTIVE_FRAMES = 4  # must hold guard for at least this many frames to count

DUCK_DROP_THRESHOLD = 0.55        # normalized downward head movement to count as a duck
SLIP_SHIFT_THRESHOLD = 0.55       # normalized sideways head movement to count as a slip
                                   # (raised from 0.35 -- that was catching normal head bob
                                   # during regular movement, not just deliberate dodges)
MIN_GAP_FRAMES = 6                # minimum frames between two separate duck/slip events


def _avg_point(kpts, indices):
    pts = [kpts[i][:2] for i in indices if kpts[i][2] > KEYPOINT_CONF_THRESHOLD]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _torso_height(kpts):
    shoulder_c = _avg_point(kpts, [LEFT_SHOULDER, RIGHT_SHOULDER])
    hip_c = _avg_point(kpts, [LEFT_HIP, RIGHT_HIP])
    if shoulder_c is None or hip_c is None:
        return None
    return float(np.linalg.norm(shoulder_c - hip_c)) + 1e-6


def detect_guard_periods(frame_series: dict) -> list:
    """Returns list of {start_frame, end_frame} for sustained guard-up periods."""
    frame_indices = sorted(frame_series.keys())
    guarding_frames = set()

    for fi in frame_indices:
        kpts = frame_series[fi]["keypoints"]
        nose = kpts[NOSE][:2] if kpts[NOSE][2] > KEYPOINT_CONF_THRESHOLD else None
        torso_h = _torso_height(kpts)
        if nose is None or torso_h is None:
            continue

        left_wrist = kpts[LEFT_WRIST]
        right_wrist = kpts[RIGHT_WRIST]

        wrists_up = 0
        if left_wrist[2] > KEYPOINT_CONF_THRESHOLD:
            if np.linalg.norm(left_wrist[:2] - nose) / torso_h < GUARD_DISTANCE_THRESHOLD:
                wrists_up += 1
        if right_wrist[2] > KEYPOINT_CONF_THRESHOLD:
            if np.linalg.norm(right_wrist[:2] - nose) / torso_h < GUARD_DISTANCE_THRESHOLD:
                wrists_up += 1

        if wrists_up == 2:  # both hands up near the head
            guarding_frames.add(fi)

    # Group consecutive frame indices into periods
    periods = []
    sorted_frames = sorted(guarding_frames)
    if not sorted_frames:
        return periods

    start = sorted_frames[0]
    prev = sorted_frames[0]
    for fi in sorted_frames[1:]:
        if fi - prev > 2:  # allow tiny 1-2 frame gaps (keypoint jitter) without breaking the period
            if prev - start + 1 >= GUARD_MIN_CONSECUTIVE_FRAMES:
                periods.append({"start_frame": start, "end_frame": prev})
            start = fi
        prev = fi
    if prev - start + 1 >= GUARD_MIN_CONSECUTIVE_FRAMES:
        periods.append({"start_frame": start, "end_frame": prev})

    return periods


def _head_deviation_series(frame_series: dict) -> list:
    """Returns [(frame_idx, vertical_drop, horizontal_shift), ...] -- the raw
    head-movement-relative-to-baseline values, WITHOUT applying any threshold.
    Used both for actual event detection and for auto-calculating thresholds
    from this video's own data."""
    frame_indices = sorted(frame_series.keys())
    baseline_window = []
    series = []

    for fi in frame_indices:
        kpts = frame_series[fi]["keypoints"]
        nose = kpts[NOSE][:2] if kpts[NOSE][2] > KEYPOINT_CONF_THRESHOLD else None
        hip_c = _avg_point(kpts, [LEFT_HIP, RIGHT_HIP])
        torso_h = _torso_height(kpts)

        if nose is None or hip_c is None or torso_h is None:
            continue

        rel_pos = (nose - hip_c) / torso_h
        baseline_window.append(rel_pos)
        if len(baseline_window) > 15:
            baseline_window.pop(0)
        if len(baseline_window) < 5:
            continue

        baseline = np.mean(baseline_window[:-1], axis=0)
        vertical_drop = rel_pos[1] - baseline[1]
        horizontal_shift = rel_pos[0] - baseline[0]
        series.append((fi, vertical_drop, horizontal_shift))

    return series


def compute_adaptive_defensive_thresholds(frame_series_by_identity: dict, percentile: float = 90,
                                           min_threshold: float = 0.25, max_threshold: float = 1.0) -> dict:
    """
    Auto-calculates DUCK_DROP_THRESHOLD and SLIP_SHIFT_THRESHOLD from this
    video's own head-movement data, instead of using one fixed number for
    every video (different videos have very different normal movement
    scales -- a fast-paced match vs. calm shadowboxing look very different).

    Returns {"duck": threshold, "slip": threshold}.
    """
    all_vertical = []
    all_horizontal = []
    for identity, frame_series in frame_series_by_identity.items():
        for fi, v_drop, h_shift in _head_deviation_series(frame_series):
            all_vertical.append(abs(v_drop))
            all_horizontal.append(abs(h_shift))

    duck_threshold = DUCK_DROP_THRESHOLD
    slip_threshold = SLIP_SHIFT_THRESHOLD
    if all_vertical:
        duck_threshold = float(np.clip(np.percentile(all_vertical, percentile), min_threshold, max_threshold))
    if all_horizontal:
        slip_threshold = float(np.clip(np.percentile(all_horizontal, percentile), min_threshold, max_threshold))

    return {"duck": duck_threshold, "slip": slip_threshold}


def detect_ducks_and_slips(frame_series: dict, duck_threshold: float = None, slip_threshold: float = None) -> list:
    """Returns list of {frame_idx, action_type ('Duck'/'Slip'), magnitude}.
    duck_threshold/slip_threshold: if None, falls back to the module-level
    constants (useful for quick standalone testing). Normally pass in values
    from compute_adaptive_defensive_thresholds() instead."""
    if duck_threshold is None:
        duck_threshold = DUCK_DROP_THRESHOLD
    if slip_threshold is None:
        slip_threshold = SLIP_SHIFT_THRESHOLD

    events = []
    last_event_frame = -MIN_GAP_FRAMES

    for fi, vertical_drop, horizontal_shift in _head_deviation_series(frame_series):
        if (fi - last_event_frame) < MIN_GAP_FRAMES:
            continue

        if vertical_drop > duck_threshold and vertical_drop > abs(horizontal_shift):
            events.append({"frame_idx": fi, "action_type": "Duck", "magnitude": round(float(vertical_drop), 3)})
            last_event_frame = fi
        elif abs(horizontal_shift) > slip_threshold and abs(horizontal_shift) > vertical_drop:
            direction = "left" if horizontal_shift < 0 else "right"
            events.append({"frame_idx": fi, "action_type": f"Slip ({direction})",
                            "magnitude": round(float(abs(horizontal_shift)), 3)})
            last_event_frame = fi

    return events


def analyze_defensive_actions(frame_series_by_identity: dict) -> dict:
    """Returns {identity: {"guard_periods": [...], "duck_slip_events": [...]}}"""
    thresholds = compute_adaptive_defensive_thresholds(frame_series_by_identity)
    results = {}
    for identity, frame_series in frame_series_by_identity.items():
        results[identity] = {
            "guard_periods": detect_guard_periods(frame_series),
            "duck_slip_events": detect_ducks_and_slips(
                frame_series, duck_threshold=thresholds["duck"], slip_threshold=thresholds["slip"]
            ),
        }
    return results


if __name__ == "__main__":
    import argparse
    from glove_confirmed_tracker import run_glove_confirmed_tracking
    from reid_fighter_tracker import compute_appearance_signatures, cluster_into_two_identities
    from strike_classifier import build_identity_frame_series

    parser = argparse.ArgumentParser(description="Phase 5: defensive action detection (standalone test)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--pose_model", default="yolov8n-pose.pt")
    parser.add_argument("--glove_model", default="models/glove_detector/weights/best.pt")
    args = parser.parse_args()

    print("Running pipeline through pose tracking + re-identification...")
    history, per_id_stats = run_glove_confirmed_tracking(
        args.video, pose_model_path=args.pose_model, glove_model_path=args.glove_model
    )
    signatures = compute_appearance_signatures(args.video, history)
    track_id_to_identity, identity_stats = cluster_into_two_identities(per_id_stats, signatures)
    frame_series_by_identity = build_identity_frame_series(history, track_id_to_identity)

    print("Detecting defensive actions...")
    thresholds = compute_adaptive_defensive_thresholds(frame_series_by_identity)
    print(f"Auto-calculated thresholds for this video: duck={thresholds['duck']:.3f}, "
          f"slip={thresholds['slip']:.3f}\n")
    results = analyze_defensive_actions(frame_series_by_identity)

    for identity, data in results.items():
        print(f"\nFighter {identity}:")
        print(f"  Guard periods: {len(data['guard_periods'])}")
        for p in data["guard_periods"]:
            print(f"    frames {p['start_frame']}-{p['end_frame']} "
                  f"({p['end_frame'] - p['start_frame'] + 1} frames)")
        print(f"  Duck/Slip events: {len(data['duck_slip_events'])}")
        for e in data["duck_slip_events"]:
            print(f"    frame {e['frame_idx']}: {e['action_type']} (magnitude {e['magnitude']})")
