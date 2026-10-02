"""
Phase 2 (heuristic version): Identify the 2 real fighters among all
tracked IDs -- WITHOUT a trained glove detector.

Uses three signals computed from Phase 1's pose-tracking data:
  1. Persistence  -- how many frames this ID appears in
                     (fighters are on-screen almost the whole clip)
  2. Movement     -- total distance the ID's body moved across the clip
                     (fighters move constantly; refs mostly stand still)
  3. Centering    -- how close to the ring center this ID stays on average
                     (fighters fight in the middle; refs/cornermen hover
                     near edges)

Each ID gets a weighted score from these three signals. The top-2
scoring IDs are kept as "confirmed fighters"; everything else
(referee, cornermen, tracking glitches) is filtered out.

NOTE: This is an approximation, not a replacement for a properly
trained glove detector. It can misfire on a very still fighter or a
very active referee. It also does NOT fix ID-swapping during
clinches (that's a separate re-identification problem) -- it only
solves "who counts as a fighter at all".
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

from pose_tracker import FighterPoseTracker, SKELETON_EDGES


# Weights for combining the three signals into one score.
# Persistence matters most (a glitchy ID with 3 frames is obviously noise),
# movement second, centering least (least reliable signal on its own).
WEIGHTS = {
    "persistence": 0.5,
    "movement": 0.3,
    "centering": 0.2,
}


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def collect_tracking_history(tracker: FighterPoseTracker, video_path: str):
    """Runs Phase 1 tracking and buffers every frame's results in memory.
    Fine for short clips (a few thousand frames); for very long videos
    you'd want to stream/aggregate instead of holding it all at once."""
    history = []
    for frame_data in tracker.track_video(video_path, save_path=None):
        history.append(frame_data)
    return history


def score_track_ids(history: list, frame_width: int, frame_height: int) -> dict:
    """Computes persistence, movement, and centering for every track ID
    seen across the clip, normalizes each 0-1, and combines into one score."""

    positions_by_id = defaultdict(list)   # track_id -> list of (frame_idx, center)
    for frame_data in history:
        for fighter in frame_data["fighters"]:
            center = _bbox_center(fighter["bbox"])
            positions_by_id[fighter["track_id"]].append(
                (frame_data["frame_idx"], center)
            )

    frame_center = np.array([frame_width / 2, frame_height / 2])
    max_possible_dist = np.linalg.norm(frame_center)  # corner-to-center distance

    raw_stats = {}
    for track_id, entries in positions_by_id.items():
        entries.sort(key=lambda e: e[0])  # sort by frame_idx
        centers = [e[1] for e in entries]

        persistence = len(entries)  # frame count

        movement = 0.0
        for i in range(1, len(centers)):
            movement += np.linalg.norm(centers[i] - centers[i - 1])

        avg_center = np.mean(centers, axis=0)
        dist_from_ring_center = np.linalg.norm(avg_center - frame_center)
        # Invert + normalize so "closer to center" = higher score
        centering = 1.0 - min(dist_from_ring_center / max_possible_dist, 1.0)

        raw_stats[track_id] = {
            "persistence": persistence,
            "movement": movement,
            "centering": centering,
        }

    # Normalize persistence and movement to 0-1 across all IDs seen
    # (centering is already 0-1 from the calculation above)
    max_persistence = max((s["persistence"] for s in raw_stats.values()), default=1)
    max_movement = max((s["movement"] for s in raw_stats.values()), default=1) or 1

    scored = {}
    for track_id, s in raw_stats.items():
        norm_persistence = s["persistence"] / max_persistence
        norm_movement = s["movement"] / max_movement
        score = (
            WEIGHTS["persistence"] * norm_persistence
            + WEIGHTS["movement"] * norm_movement
            + WEIGHTS["centering"] * s["centering"]
        )
        scored[track_id] = {
            **s,
            "norm_persistence": norm_persistence,
            "norm_movement": norm_movement,
            "score": score,
        }

    return scored


def select_fighters(scored: dict, top_n: int = 2) -> set:
    """Returns the set of track_ids judged to be real fighters."""
    ranked = sorted(scored.items(), key=lambda kv: kv[1]["score"], reverse=True)
    return {track_id for track_id, _ in ranked[:top_n]}


def render_filtered_video(
    video_path: str,
    history: list,
    confirmed_ids: set,
    out_path: str,
):
    """Re-draws the video, showing confirmed fighters in green with their
    skeleton, and filtered-out IDs in gray (so you can visually verify
    the heuristic worked) with a small 'filtered' label."""

    cap = cv2.VideoCapture(video_path)
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
            frame_data = history[frame_idx]
            for fighter in frame_data["fighters"]:
                is_fighter = fighter["track_id"] in confirmed_ids
                color = (0, 200, 0) if is_fighter else (120, 120, 120)  # BGR
                label = f"Fighter {fighter['track_id']}" if is_fighter else f"filtered ({fighter['track_id']})"

                x1, y1, x2, y2 = [int(v) for v in fighter["bbox"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                kpts = fighter["keypoints"]
                for a, b in SKELETON_EDGES:
                    if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:  # confidence check
                        pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                        pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                        cv2.line(frame, pt1, pt2, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2 (heuristic): filter tracked IDs down to the real fighters"
    )
    parser.add_argument("--video", required=True, help="Path to input boxing video")
    parser.add_argument("--out", default="data/output/filtered.mp4", help="Path to save filtered output")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model to use")
    parser.add_argument("--top_n", type=int, default=2, help="How many IDs to keep as fighters")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tracker = FighterPoseTracker(model_path=args.model)

    print("Pass 1/2: running pose tracking and collecting history...")
    history = collect_tracking_history(tracker, args.video)

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("Scoring track IDs (persistence + movement + centering)...")
    scored = score_track_ids(history, width, height)

    print("\nTrack ID scores:")
    for track_id, s in sorted(scored.items(), key=lambda kv: kv[1]["score"], reverse=True):
        print(f"  ID {track_id:>3}: score={s['score']:.3f}  "
              f"(frames={s['persistence']}, movement={s['movement']:.0f}px, "
              f"centering={s['centering']:.2f})")

    confirmed = select_fighters(scored, top_n=args.top_n)
    print(f"\nConfirmed fighter IDs: {sorted(confirmed)}")

    print("\nPass 2/2: rendering filtered output video...")
    render_filtered_video(args.video, history, confirmed, args.out)

    print(f"\nDone. Filtered video saved to: {args.out}")
    print("Green = confirmed fighter, Gray = filtered out (likely ref/cornerman/noise).")
    print("Watch the output -- if it misidentified anyone, we can adjust the WEIGHTS "
          "in this script or fall back to training a real glove detector later.")
