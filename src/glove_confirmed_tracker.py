"""
Phase 2b: Confirm real fighters using the trained glove detector
--------------------------------------------------------------------
Combines two models per frame:
  1. Pose model (Phase 1)          -> skeleton + track ID per person
  2. Glove detector (Phase 2, ours)-> bounding boxes of real gloves

For each tracked person, we check whether a detected glove sits near
either wrist keypoint. A track ID that has gloves confirmed near its
wrists across most of its frames is a real fighter; one that rarely
or never does (referee, cornerman, tracking noise) gets filtered out.

This replaces the movement/position heuristic from fighter_filter.py
with actual visual evidence -- much more reliable, as we saw the
heuristic mistake the referee for a fighter in testing.
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

from pose_tracker import SKELETON_EDGES

LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10

WRIST_CONF_THRESHOLD = 0.3     # ignore low-confidence wrist keypoints
GLOVE_MATCH_PADDING = 25       # pixels of extra tolerance around each glove box
MIN_CONFIRMATION_RATE = 0.15   # a track ID needs gloves detected in at least
                                # this fraction of its frames to be considered


def _wrist_in_any_glove(wrist_xy, glove_boxes, padding=GLOVE_MATCH_PADDING):
    wx, wy = wrist_xy
    for (gx1, gy1, gx2, gy2) in glove_boxes:
        if (gx1 - padding) <= wx <= (gx2 + padding) and (gy1 - padding) <= wy <= (gy2 + padding):
            return True
    return False


def run_glove_confirmed_tracking(
    video_path: str,
    pose_model_path: str = "yolov8n-pose.pt",
    glove_model_path: str = "models/glove_detector/weights/best.pt",
    conf: float = 0.4,
):
    """
    Runs pose tracking + glove detection together, frame by frame.
    Returns (history, per_id_stats) where:
      history: list of frame dicts (frame_idx, fighters[track_id, bbox, keypoints])
               -- same shape as pose_tracker.py's output, for reuse in rendering
      per_id_stats: dict of track_id -> {frames_seen, glove_confirmed_frames, confirmation_rate}
    """
    pose_model = YOLO(pose_model_path)
    glove_model = YOLO(glove_model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    history = []
    frames_seen = defaultdict(int)
    glove_confirmed = defaultdict(int)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_result = pose_model.track(
            frame, persist=True, tracker="botsort.yaml", conf=conf, verbose=False
        )[0]

        glove_result = glove_model(frame, conf=conf, verbose=False)[0]
        glove_boxes = (
            glove_result.boxes.xyxy.cpu().numpy()
            if glove_result.boxes is not None and len(glove_result.boxes) > 0
            else np.empty((0, 4))
        )

        frame_data = {"frame_idx": frame_idx, "fighters": []}

        if pose_result.boxes is not None and pose_result.boxes.id is not None:
            ids = pose_result.boxes.id.cpu().numpy().astype(int)
            boxes = pose_result.boxes.xyxy.cpu().numpy()
            keypoints = pose_result.keypoints.data.cpu().numpy()

            for track_id, bbox, kpts in zip(ids, boxes, keypoints):
                frame_data["fighters"].append({
                    "track_id": int(track_id),
                    "bbox": bbox.tolist(),
                    "keypoints": kpts,
                })

                frames_seen[int(track_id)] += 1

                left_wrist = kpts[LEFT_WRIST_IDX]
                right_wrist = kpts[RIGHT_WRIST_IDX]
                confirmed = False
                if left_wrist[2] > WRIST_CONF_THRESHOLD and _wrist_in_any_glove(left_wrist[:2], glove_boxes):
                    confirmed = True
                if right_wrist[2] > WRIST_CONF_THRESHOLD and _wrist_in_any_glove(right_wrist[:2], glove_boxes):
                    confirmed = True

                if confirmed:
                    glove_confirmed[int(track_id)] += 1

        history.append(frame_data)
        frame_idx += 1

    cap.release()

    per_id_stats = {}
    for track_id, seen in frames_seen.items():
        confirmed_count = glove_confirmed[track_id]
        per_id_stats[track_id] = {
            "frames_seen": seen,
            "glove_confirmed_frames": confirmed_count,
            "confirmation_rate": confirmed_count / seen if seen > 0 else 0.0,
        }

    return history, per_id_stats


def select_fighters_by_glove(per_id_stats: dict, top_n: int = 2) -> set:
    """Ranks track IDs by (confirmation_rate * frames_seen) -- rewards IDs
    that are both persistent AND consistently show real gloves -- and
    keeps the top N above a minimum confirmation rate."""
    candidates = {
        tid: s for tid, s in per_id_stats.items()
        if s["confirmation_rate"] >= MIN_CONFIRMATION_RATE
    }
    ranked = sorted(
        candidates.items(),
        key=lambda kv: kv[1]["confirmation_rate"] * kv[1]["frames_seen"],
        reverse=True,
    )
    return {tid for tid, _ in ranked[:top_n]}


def render_glove_confirmed_video(video_path: str, history: list, confirmed_ids: set, out_path: str):
    """Same rendering style as fighter_filter.py: green = confirmed fighter,
    gray = filtered out."""
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
                is_fighter = fighter["track_id"] in confirmed_ids
                color = (0, 200, 0) if is_fighter else (120, 120, 120)
                label = f"Fighter {fighter['track_id']}" if is_fighter else f"filtered ({fighter['track_id']})"

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

    parser = argparse.ArgumentParser(description="Phase 2b: glove-confirmed fighter filtering")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="data/output/glove_filtered.mp4")
    parser.add_argument("--pose_model", default="yolov8n-pose.pt")
    parser.add_argument("--glove_model", default="models/glove_detector/weights/best.pt")
    parser.add_argument("--top_n", type=int, default=2)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("Running pose tracking + glove detection together (this is slower than "
          "Phase 1 alone, since it runs two models per frame)...")
    history, stats = run_glove_confirmed_tracking(
        args.video, pose_model_path=args.pose_model, glove_model_path=args.glove_model
    )

    print("\nTrack ID glove-confirmation stats:")
    for tid, s in sorted(stats.items(), key=lambda kv: kv[1]["confirmation_rate"] * kv[1]["frames_seen"], reverse=True):
        print(f"  ID {tid:>3}: frames={s['frames_seen']:>4}  "
              f"glove_confirmed={s['glove_confirmed_frames']:>4}  "
              f"rate={s['confirmation_rate']:.2f}")

    confirmed = select_fighters_by_glove(stats, top_n=args.top_n)
    print(f"\nConfirmed fighter IDs (via real glove detection): {sorted(confirmed)}")

    print("\nRendering output video...")
    render_glove_confirmed_video(args.video, history, confirmed, args.out)

    print(f"\nDone. Saved to: {args.out}")
    print("Green = confirmed fighter (gloves detected), Gray = filtered out.")
