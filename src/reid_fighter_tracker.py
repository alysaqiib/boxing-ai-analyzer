"""
Phase 2c: Appearance-based re-identification
------------------------------------------------
Problem: BoT-SORT (Phase 1) sometimes loses a fighter during a clinch
or occlusion and assigns a NEW track ID when they reappear, even
though it's the same physical person. This fragments one fighter's
data across several IDs, which breaks the glove-confirmation stats
(each fragment individually may look like "noise") and would later
break strike attribution ("who threw that punch?").

Fix: give every track ID a visual signature (a color histogram of
their bounding-box region -- captures gear color, skin tone, shorts,
etc). Track IDs with very similar signatures almost certainly belong
to the same physical fighter, even if BoT-SORT assigned them
different numbers. We cluster all track IDs into exactly 2 persistent
identities ("Fighter A" / "Fighter B") using this signature, then
merge each identity's stats before doing glove-confirmation filtering.

This does NOT require any training or dataset -- it's classic
appearance matching using OpenCV histograms.
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

from pose_tracker import SKELETON_EDGES
from glove_confirmed_tracker import (
    run_glove_confirmed_tracking,
    MIN_CONFIRMATION_RATE,
)

MIN_FRAMES_TO_CONSIDER = 5   # ignore track IDs that barely appear (tracking noise)
HIST_BINS = 32               # color histogram resolution per channel

# A track ID with at least this many frames AND a glove-confirmation rate
# at or below this threshold is almost certainly NOT a fighter (referee/
# cornerman) -- exclude it from appearance clustering entirely. Using a
# low-rate threshold rather than exactly zero, since a referee's green
# mitts can occasionally trigger a stray false-positive glove detection.
DEFINITE_NON_FIGHTER_MIN_FRAMES = 15
DEFINITE_NON_FIGHTER_MAX_RATE = 0.05


def _crop_safe(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _compute_histogram(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS, HIST_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _similarity(hist_a, hist_b):
    """Returns a similarity score: higher = more visually similar."""
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


def compute_appearance_signatures(video_path: str, history: list) -> dict:
    """Returns track_id -> averaged color histogram across all frames that ID appears in."""
    cap = cv2.VideoCapture(str(video_path))
    accum = defaultdict(list)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(history):
            for fighter in history[frame_idx]["fighters"]:
                crop = _crop_safe(frame, fighter["bbox"])
                if crop is not None and crop.size > 0:
                    accum[fighter["track_id"]].append(_compute_histogram(crop))
        frame_idx += 1

    cap.release()

    signatures = {}
    for track_id, hists in accum.items():
        avg_hist = np.mean(hists, axis=0).astype(np.float32)
        signatures[track_id] = avg_hist

    return signatures


def cluster_into_two_identities(per_id_stats: dict, signatures: dict):
    """
    Groups all track IDs into at most 2 persistent identities based on
    appearance similarity. Returns (track_id_to_identity, identity_stats).

    track_id_to_identity: dict track_id -> 'A' / 'B' / None (None = filtered as noise)
    identity_stats: dict 'A'/'B' -> merged {frames_seen, glove_confirmed_frames, confirmation_rate}
    """
    # Step 1: hard-exclude track IDs that are almost certainly NOT fighters --
    # persistent on screen, but never once had a glove confirmed nearby.
    # (This is what the referee looks like: high frame count, zero evidence.)
    definite_non_fighters = {
        tid for tid, s in per_id_stats.items()
        if s["frames_seen"] >= DEFINITE_NON_FIGHTER_MIN_FRAMES
        and s["confirmation_rate"] <= DEFINITE_NON_FIGHTER_MAX_RATE
    }

    candidates = {
        tid: s for tid, s in per_id_stats.items()
        if s["frames_seen"] >= MIN_FRAMES_TO_CONSIDER
        and tid in signatures
        and tid not in definite_non_fighters
    }

    if not candidates:
        return {tid: None for tid in per_id_stats}, {}

    # Seed identity A = the candidate with the STRONGEST glove evidence
    # (not just the most persistent -- that's what caused the referee bug)
    sorted_by_evidence = sorted(
        candidates.items(),
        key=lambda kv: (kv[1]["glove_confirmed_frames"], kv[1]["frames_seen"]),
        reverse=True,
    )
    seed_a_id = sorted_by_evidence[0][0]

    # Seed identity B = the most VISUALLY DIFFERENT track ID among the
    # candidates with the next-strongest glove evidence (avoids seeding
    # on a tiny noise blob that just happens to look different)
    top_candidates = [tid for tid, _ in sorted_by_evidence[:8]]  # up to 8 strongest fragments
    seed_b_id = None
    lowest_similarity = float("inf")
    for tid in top_candidates:
        if tid == seed_a_id:
            continue
        sim = _similarity(signatures[seed_a_id], signatures[tid])
        if sim < lowest_similarity:
            lowest_similarity = sim
            seed_b_id = tid

    if seed_b_id is None:
        # Only one real fighter fragment found -- everything maps to A
        track_id_to_identity = {tid: "A" for tid in candidates}
    else:
        track_id_to_identity = {}
        for tid in candidates:
            sim_a = _similarity(signatures[seed_a_id], signatures[tid])
            sim_b = _similarity(signatures[seed_b_id], signatures[tid])
            track_id_to_identity[tid] = "A" if sim_a >= sim_b else "B"

    # Everything hard-excluded (referee) or too small to matter stays filtered
    for tid in per_id_stats:
        if tid not in track_id_to_identity:
            track_id_to_identity[tid] = None

    # Merge stats per identity (skip filtered/None -- not a real identity)
    identity_stats = defaultdict(lambda: {"frames_seen": 0, "glove_confirmed_frames": 0})
    for tid, identity in track_id_to_identity.items():
        if identity is None:
            continue
        identity_stats[identity]["frames_seen"] += per_id_stats[tid]["frames_seen"]
        identity_stats[identity]["glove_confirmed_frames"] += per_id_stats[tid]["glove_confirmed_frames"]

    for identity, s in identity_stats.items():
        s["confirmation_rate"] = s["glove_confirmed_frames"] / s["frames_seen"] if s["frames_seen"] > 0 else 0.0

    return track_id_to_identity, dict(identity_stats)


IDENTITY_COLORS = {
    "A": (0, 200, 0),     # green (BGR: no blue, high green, no red)
    "B": (0, 140, 255),   # orange (BGR: no blue, mid green, high red)
    None: (120, 120, 120) # gray -- filtered/noise
}


def render_reid_video(video_path: str, history: list, track_id_to_identity: dict, out_path: str):
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

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(history):
            for fighter in history[frame_idx]["fighters"]:
                identity = track_id_to_identity.get(fighter["track_id"])
                color = IDENTITY_COLORS[identity]
                label = f"Fighter {identity}" if identity else f"filtered ({fighter['track_id']})"

                x1, y1, x2, y2 = [int(v) for v in fighter["bbox"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                kpts = fighter["keypoints"]
                for a, b in SKELETON_EDGES:
                    if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
                        pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                        pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                        cv2.line(frame, pt1, pt2, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2c: appearance-based re-identification")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="data/output/reid_filtered.mp4")
    parser.add_argument("--pose_model", default="yolov8n-pose.pt")
    parser.add_argument("--glove_model", default="models/glove_detector/weights/best.pt")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("Step 1/3: running pose tracking + glove detection...")
    history, per_id_stats = run_glove_confirmed_tracking(
        args.video, pose_model_path=args.pose_model, glove_model_path=args.glove_model
    )

    print("Step 2/3: computing appearance signatures and clustering into 2 identities...")
    signatures = compute_appearance_signatures(args.video, history)
    track_id_to_identity, identity_stats = cluster_into_two_identities(per_id_stats, signatures)

    print("\nRaw track ID -> merged identity mapping:")
    for tid, identity in sorted(track_id_to_identity.items()):
        print(f"  ID {tid:>3} -> Fighter {identity}")

    print("\nMerged identity stats:")
    for identity, s in identity_stats.items():
        print(f"  Fighter {identity}: frames={s['frames_seen']}, "
              f"glove_confirmed={s['glove_confirmed_frames']}, rate={s['confirmation_rate']:.2f}")

    print("\nStep 3/3: rendering output video...")
    render_reid_video(args.video, history, track_id_to_identity, args.out)

    print(f"\nDone. Saved to: {args.out}")
    print("Green = Fighter A, Orange = Fighter B, Gray = filtered (referee/noise).")
    print("Both colors should now stay CONSISTENT on the same physical fighter throughout the clip, "
          "even if the underlying track ID changed during a clinch.")
