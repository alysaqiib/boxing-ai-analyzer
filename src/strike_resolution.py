"""
Phase 4: Landed / Blocked / Missed resolution
--------------------------------------------------
For each detected strike (from strike_classifier.py), checks where the
OPPONENT actually was in that exact frame, and classifies the strike as:

  LANDED  -- striking wrist ended up close to the opponent's head, body,
             or legs
  BLOCKED -- striking wrist ended up closer to the opponent's own guard
             (their wrists/forearms) than to an open target
  MISSED  -- striking wrist wasn't close to either

HONEST LIMITATIONS
  - This is 2D proximity on a single camera angle -- there's no real
    depth information, so "close in the image" isn't always "actually
    touching in 3D". A punch that LOOKS close because of camera angle
    but was actually a foot in front of/behind the opponent will be
    misjudged. Treat this as a reasonable estimate, not ground truth.
  - Distance thresholds are defaults, not tuned against your footage --
    expect to adjust LANDED_THRESHOLD after watching real output.
"""

import numpy as np
from collections import defaultdict

from strike_classifier import LEFT_WRIST, RIGHT_WRIST

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

KEYPOINT_CONF_THRESHOLD = 0.3

# Normalized distance (relative to opponent's bbox diagonal) under which
# something counts as "close enough". Tuned up slightly from a real test
# case (a punch that visibly reached the opponent's guard was scoring just
# above the original 0.30 cutoff, likely due to keypoint uncertainty when
# hands are close together and briefly overlapping).
LANDED_THRESHOLD = 0.38


def _bbox_diag(bbox):
    x1, y1, x2, y2 = bbox
    return float(np.hypot(x2 - x1, y2 - y1)) + 1e-6


def _avg_point(kpts, indices):
    """Average position of a set of keypoints, skipping low-confidence ones."""
    pts = [kpts[i][:2] for i in indices if kpts[i][2] > KEYPOINT_CONF_THRESHOLD]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _find_fighter_entry(frame_data, track_id_to_identity, identity):
    for fighter in frame_data["fighters"]:
        if track_id_to_identity.get(fighter["track_id"]) == identity:
            return fighter
    return None


def _closest_zone_distance(strike: dict, striker_identity: str, opponent_identity: str,
                            history: list, track_id_to_identity: dict):
    """Returns (closest_zone_dist, closest_zone_name, guard_dist) for one strike,
    or (None, None, None) if it can't be computed (opponent not visible, etc).
    Shared by resolve_strike() and the adaptive-threshold calculator below."""
    frame_idx = strike["frame_idx"]
    if frame_idx >= len(history):
        return None, None, None

    frame_data = history[frame_idx]
    striker_entry = _find_fighter_entry(frame_data, track_id_to_identity, striker_identity)
    opponent_entry = _find_fighter_entry(frame_data, track_id_to_identity, opponent_identity)
    if striker_entry is None or opponent_entry is None:
        return None, None, None

    wrist_idx = LEFT_WRIST if strike["side"] == "left" else RIGHT_WRIST
    striker_wrist = np.array(striker_entry["keypoints"][wrist_idx][:2])

    opp_kpts = opponent_entry["keypoints"]
    diag = _bbox_diag(opponent_entry["bbox"])

    zones = {
        "Head": _avg_point(opp_kpts, [NOSE]),
        "Body": _avg_point(opp_kpts, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]),
        "Leg": _avg_point(opp_kpts, [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]),
    }
    guard_points = [
        p for p in [
            opp_kpts[LEFT_WRIST][:2] if opp_kpts[LEFT_WRIST][2] > KEYPOINT_CONF_THRESHOLD else None,
            opp_kpts[RIGHT_WRIST][:2] if opp_kpts[RIGHT_WRIST][2] > KEYPOINT_CONF_THRESHOLD else None,
        ] if p is not None
    ]

    zone_dists = {}
    for zone_name, zone_pos in zones.items():
        if zone_pos is not None:
            zone_dists[zone_name] = np.linalg.norm(striker_wrist - zone_pos) / diag

    if not zone_dists:
        return None, None, None

    guard_dist = None
    if guard_points:
        guard_dist = min(np.linalg.norm(striker_wrist - np.array(g)) / diag for g in guard_points)

    closest_zone, closest_zone_dist = min(zone_dists.items(), key=lambda kv: kv[1])
    return closest_zone_dist, closest_zone, guard_dist


def compute_adaptive_landed_threshold(strikes_by_identity: dict, history: list, track_id_to_identity: dict,
                                       percentile: float = 40, min_threshold: float = 0.20,
                                       max_threshold: float = 0.55) -> float:
    """
    Auto-calculates LANDED_THRESHOLD from this video's own data, instead of a
    fixed guess. Looks at how close every detected strike actually got to the
    opponent across the whole video, and sets the cutoff at a percentile of
    that distribution -- so a video where fighters generally fight at closer
    or longer range automatically gets a matching threshold.

    HONEST CAVEAT: this is less rigorously justified than the strike-speed
    adaptive threshold (there's no clean "normal vs. spike" separation here
    the way there is for movement speed) -- it's a reasonable heuristic, not
    a guaranteed-correct calibration. Still likely better than one fixed
    number used for every video regardless of typical fighting range.
    """
    identities = list(strikes_by_identity.keys())
    if len(identities) != 2:
        return LANDED_THRESHOLD

    id_a, id_b = identities
    all_dists = []
    for striker, opponent in [(id_a, id_b), (id_b, id_a)]:
        for strike in strikes_by_identity[striker]:
            dist, _, _ = _closest_zone_distance(strike, striker, opponent, history, track_id_to_identity)
            if dist is not None:
                all_dists.append(dist)

    if not all_dists:
        return LANDED_THRESHOLD

    return float(np.clip(np.percentile(all_dists, percentile), min_threshold, max_threshold))


def resolve_strike(strike: dict, striker_identity: str, opponent_identity: str,
                    history: list, track_id_to_identity: dict, landed_threshold: float = None) -> dict:
    """
    Returns the strike dict with two new fields added:
      "result": "Landed" / "Blocked" / "Missed" / "Unknown" (opponent not visible)
      "target_zone": "Head" / "Body" / "Leg" / None
    landed_threshold: if None, falls back to the module-level LANDED_THRESHOLD
                       constant. Normally pass a value from
                       compute_adaptive_landed_threshold() instead.
    """
    if landed_threshold is None:
        landed_threshold = LANDED_THRESHOLD

    result = dict(strike)
    result["result"] = "Unknown"
    result["target_zone"] = None

    closest_zone_dist, closest_zone, guard_dist = _closest_zone_distance(
        strike, striker_identity, opponent_identity, history, track_id_to_identity
    )
    if closest_zone_dist is None:
        return result

    if guard_dist is not None and guard_dist < closest_zone_dist and guard_dist < landed_threshold:
        result["result"] = "Blocked"
    elif closest_zone_dist < landed_threshold:
        result["result"] = "Landed"
        result["target_zone"] = closest_zone
    else:
        result["result"] = "Missed"

    return result


def resolve_all_strikes(strikes_by_identity: dict, history: list, track_id_to_identity: dict,
                         landed_threshold: float = None) -> dict:
    """
    strikes_by_identity: {"A": [...strikes], "B": [...strikes]}
    landed_threshold: if None, auto-calculates from this video's own data via
                       compute_adaptive_landed_threshold().
    Returns the same structure with each strike enriched with "result" and "target_zone".
    Assumes exactly 2 identities (the opponent of A is B and vice versa).
    """
    identities = list(strikes_by_identity.keys())
    if len(identities) != 2:
        return strikes_by_identity

    if landed_threshold is None:
        landed_threshold = compute_adaptive_landed_threshold(strikes_by_identity, history, track_id_to_identity)

    id_a, id_b = identities
    resolved = {id_a: [], id_b: []}

    for strike in strikes_by_identity[id_a]:
        resolved[id_a].append(resolve_strike(strike, id_a, id_b, history, track_id_to_identity, landed_threshold))
    for strike in strikes_by_identity[id_b]:
        resolved[id_b].append(resolve_strike(strike, id_b, id_a, history, track_id_to_identity, landed_threshold))

    return resolved


def summarize_results(resolved_strikes_by_identity: dict) -> dict:
    """Returns {identity: {"Landed": n, "Blocked": n, "Missed": n, "Unknown": n}}"""
    summary = {}
    for identity, strikes in resolved_strikes_by_identity.items():
        counts = defaultdict(int)
        for s in strikes:
            counts[s["result"]] += 1
        summary[identity] = dict(counts)
    return summary


if __name__ == "__main__":
    import argparse
    from glove_confirmed_tracker import run_glove_confirmed_tracking
    from reid_fighter_tracker import compute_appearance_signatures, cluster_into_two_identities
    from strike_classifier import build_identity_frame_series, detect_strikes_for_identity

    parser = argparse.ArgumentParser(description="Phase 4: landed/blocked/missed resolution (standalone test)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--pose_model", default="yolov8n-pose.pt")
    parser.add_argument("--glove_model", default="models/glove_detector/weights/best.pt")
    args = parser.parse_args()

    print("Running full pipeline up through strike detection...")
    history, per_id_stats = run_glove_confirmed_tracking(
        args.video, pose_model_path=args.pose_model, glove_model_path=args.glove_model
    )
    signatures = compute_appearance_signatures(args.video, history)
    track_id_to_identity, identity_stats = cluster_into_two_identities(per_id_stats, signatures)
    frame_series_by_identity = build_identity_frame_series(history, track_id_to_identity)
    strikes_by_identity = {
        identity: detect_strikes_for_identity(fs)
        for identity, fs in frame_series_by_identity.items()
    }

    print("Resolving landed/blocked/missed...")
    adaptive_threshold = compute_adaptive_landed_threshold(strikes_by_identity, history, track_id_to_identity)
    print(f"Auto-calculated LANDED_THRESHOLD for this video: {adaptive_threshold:.3f}\n")
    resolved = resolve_all_strikes(strikes_by_identity, history, track_id_to_identity, adaptive_threshold)

    print("\nResults:")
    summary = summarize_results(resolved)
    for identity, counts in summary.items():
        print(f"  Fighter {identity}: {counts}")

    print("\nPer-strike detail:")
    for identity, strikes in resolved.items():
        for s in strikes:
            print(f"  Fighter {identity} frame {s['frame_idx']}: {s['punch_type']} ({s['side']}) "
                  f"-> {s['result']}" + (f" [{s['target_zone']}]" if s['target_zone'] else ""))
