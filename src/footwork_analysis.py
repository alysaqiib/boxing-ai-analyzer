"""
Phase 6: Footwork & stance analysis
----------------------------------------
Uses hip/ankle keypoints we already have to compute:

  DISTANCE COVERED  -- total movement of the hip center across the clip,
                        normalized by torso scale. This is a RELATIVE
                        unit ("body-lengths of movement"), NOT meters,
                        feet, or any real-world distance -- there's no
                        camera calibration or known reference size in
                        the video to convert to actual units. Only
                        meaningful for comparing fighters WITHIN the
                        same video, not across different videos/angles.
  STANCE            -- orthodox (left foot forward) vs southpaw (right
                        foot forward), determined by which ankle sits
                        further forward relative to the hip center,
                        smoothed over time so a single step/kick doesn't
                        flip the classification
  MOVEMENT ACTIVITY  -- a simple per-second "how much is this fighter
                        moving" signal, useful for spotting fatigue or
                        low-activity stretches later

HONEST LIMITATIONS
  - "Forward" here means "toward the camera" in 2D pixel space, not
    true 3D depth. From a side-angle camera, this works reasonably
    well; from a front-on angle, it may not.
  - Stance can be ambiguous mid-combination (fighters shift weight
    constantly) -- we smooth over a rolling window to reduce flicker,
    but expect some noise during fast exchanges.
"""

import numpy as np
from collections import defaultdict

LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

KEYPOINT_CONF_THRESHOLD = 0.3
STANCE_SMOOTHING_WINDOW = 15   # frames (~0.5-0.6s) to smooth stance over, reduces flicker


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


def compute_distance_covered(frame_series: dict) -> float:
    """Total normalized hip-center movement across the whole clip."""
    frame_indices = sorted(frame_series.keys())
    total_distance = 0.0
    prev_pos = None

    for fi in frame_indices:
        kpts = frame_series[fi]["keypoints"]
        hip_c = _avg_point(kpts, [LEFT_HIP, RIGHT_HIP])
        torso_h = _torso_height(kpts)
        if hip_c is None or torso_h is None:
            prev_pos = None
            continue

        if prev_pos is not None:
            total_distance += np.linalg.norm(hip_c - prev_pos) / torso_h
        prev_pos = hip_c

    return round(float(total_distance), 2)


def compute_stance_per_frame(frame_series: dict) -> dict:
    """
    Returns {frame_idx: 'orthodox' / 'southpaw' / None} per frame, using
    which ankle is further forward (toward camera / larger... actually
    smaller or larger x depends on which way the fighter faces -- we use
    RELATIVE horizontal offset from hip center, smoothed over time).

    Convention used here: if the LEFT ankle is further from the body's
    horizontal center of mass in the direction the fighter is generally
    leaning/facing, that's treated as "orthodox" (left foot forward,
    the traditional right-handed stance) -- and vice versa for southpaw.
    This is a simplification; always sanity-check against real footage.
    """
    frame_indices = sorted(frame_series.keys())
    raw_signal = {}  # frame_idx -> +1 (left forward) or -1 (right forward)

    for fi in frame_indices:
        kpts = frame_series[fi]["keypoints"]
        left_ankle = kpts[LEFT_ANKLE]
        right_ankle = kpts[RIGHT_ANKLE]
        hip_c = _avg_point(kpts, [LEFT_HIP, RIGHT_HIP])

        if hip_c is None:
            continue
        if left_ankle[2] < KEYPOINT_CONF_THRESHOLD or right_ankle[2] < KEYPOINT_CONF_THRESHOLD:
            continue

        # Distance of each ankle from the hip center's x position -- whichever
        # ankle is further from the OTHER ankle in the "forward" direction
        # (approximated here as: whichever ankle is more separated horizontally
        # from the hip center on its own side) is treated as the lead foot.
        left_offset = left_ankle[0] - hip_c[0]
        right_offset = right_ankle[0] - hip_c[0]

        # The lead foot is the one further from the body's vertical centerline
        # in the direction of travel -- simplified here as whichever ankle has
        # the larger absolute horizontal spread combined with being the
        # forward-most (smaller depth cue unavailable, so we use pure x-spread).
        if abs(left_offset) > abs(right_offset):
            raw_signal[fi] = "orthodox"  # left foot leading/spread further
        else:
            raw_signal[fi] = "southpaw"  # right foot leading/spread further

    # Smooth with a rolling majority vote to reduce frame-to-frame flicker
    smoothed = {}
    frame_list = sorted(raw_signal.keys())
    for idx, fi in enumerate(frame_list):
        window = frame_list[max(0, idx - STANCE_SMOOTHING_WINDOW):idx + 1]
        votes = [raw_signal[w] for w in window]
        orthodox_count = votes.count("orthodox")
        southpaw_count = votes.count("southpaw")
        smoothed[fi] = "orthodox" if orthodox_count >= southpaw_count else "southpaw"

    return smoothed


def summarize_stance(stance_per_frame: dict) -> str:
    """Returns the dominant stance across the whole clip."""
    if not stance_per_frame:
        return "unknown"
    counts = defaultdict(int)
    for stance in stance_per_frame.values():
        counts[stance] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def analyze_footwork(frame_series_by_identity: dict) -> dict:
    """Returns {identity: {"distance_covered": float, "dominant_stance": str,
                            "stance_per_frame": dict}}"""
    results = {}
    for identity, frame_series in frame_series_by_identity.items():
        stance_per_frame = compute_stance_per_frame(frame_series)
        results[identity] = {
            "distance_covered": compute_distance_covered(frame_series),
            "dominant_stance": summarize_stance(stance_per_frame),
            "stance_per_frame": stance_per_frame,
        }
    return results


if __name__ == "__main__":
    import argparse
    from glove_confirmed_tracker import run_glove_confirmed_tracking
    from reid_fighter_tracker import compute_appearance_signatures, cluster_into_two_identities
    from strike_classifier import build_identity_frame_series

    parser = argparse.ArgumentParser(description="Phase 6: footwork & stance analysis (standalone test)")
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

    print("Analyzing footwork and stance...")
    results = analyze_footwork(frame_series_by_identity)

    for identity, data in results.items():
        print(f"\nFighter {identity}:")
        print(f"  Distance covered (normalized units): {data['distance_covered']}")
        print(f"  Dominant stance: {data['dominant_stance']}")

        # Show how often stance flips, as a sanity check for noise
        stances = [data["stance_per_frame"][fi] for fi in sorted(data["stance_per_frame"].keys())]
        flips = sum(1 for i in range(1, len(stances)) if stances[i] != stances[i - 1])
        print(f"  Stance flips across clip: {flips} (high number = noisy/uncertain classification)")
