"""
Phase 1: Core pose tracking + tracking IDs (foundation)
--------------------------------------------------------
Uses a pretrained YOLOv8-Pose model + built-in BoT-SORT tracker
(via Ultralytics) to detect and track fighters in a boxing video.

This is deliberately the FIRST thing to build: everything else
(strike classification, hit-zones, footwork analysis) depends on
having stable, correctly-ID'd skeletons for both fighters across
every frame -- including through clinches and occlusion.

No custom training required for this phase. Works with any
boxing/sparring clip.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


# COCO keypoint indices (what YOLOv8-Pose gives us per person)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # arms + shoulders
    (5, 11), (6, 12), (11, 12),                    # torso
    (11, 13), (13, 15), (12, 14), (14, 16),        # legs
]


class FighterPoseTracker:
    """
    Wraps YOLOv8-Pose + BoT-SORT to produce per-frame, per-fighter
    skeleton + tracking ID data.
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt", conf: float = 0.4):
        """
        model_path: pretrained pose model. Ultralytics will auto-download
                    yolov8n-pose.pt (nano, fastest) if not present locally.
                    Swap for yolov8s/m/l-pose.pt for higher accuracy once
                    you have GPU access.
        conf:       detection confidence threshold.
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def track_video(self, video_path: str, save_path: str | None = None, show: bool = False):
        """
        Runs tracking across an entire video and yields per-frame results.

        Yields a dict per frame:
            {
                "frame_idx": int,
                "fighters": [
                    {
                        "track_id": int,
                        "bbox": [x1, y1, x2, y2],
                        "keypoints": np.ndarray shape (17, 3)  # x, y, confidence
                    },
                    ...
                ]
            }
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        writer = None
        if save_path:
            def _open_writer(path, fps, size):
                for code in ["avc1", "H264", "mp4v"]:
                    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
                    if w.isOpened():
                        return w
                    w.release()
                raise RuntimeError(f"Could not open video writer for {path}")

            writer = _open_writer(save_path, fps, (width, height))

        # `stream=True` + `.track()` gives us persistent IDs across frames
        # via BoT-SORT (Ultralytics' default tracker for pose/detection).
        results_stream = self.model.track(
            source=video_path,
            conf=self.conf,
            persist=True,
            tracker="botsort.yaml",
            stream=True,
            verbose=False,
        )

        for frame_idx, result in enumerate(results_stream):
            frame_data = {"frame_idx": frame_idx, "fighters": []}

            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.cpu().numpy()
                keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

                for track_id, bbox, kpts in zip(ids, boxes, keypoints):
                    frame_data["fighters"].append({
                        "track_id": int(track_id),
                        "bbox": bbox.tolist(),
                        "keypoints": kpts,  # (17, 3) -> x, y, conf per joint
                    })

            annotated = result.plot()  # draws skeletons + boxes + IDs

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("Boxing AI Tracker - Phase 1", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            yield frame_data

        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def get_wrist_positions(fighter: dict) -> dict:
    """Convenience helper: pull left/right wrist (x, y, conf) for later
    strike-velocity calculations in Phase 3."""
    kpts = fighter["keypoints"]
    return {
        "left_wrist": kpts[9],
        "right_wrist": kpts[10],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Fighter pose tracking test")
    parser.add_argument("--video", required=True, help="Path to input boxing video")
    parser.add_argument("--out", default="data/output/tracked.mp4", help="Path to save annotated output")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model to use")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tracker = FighterPoseTracker(model_path=args.model)

    num_frames = 0
    max_ids_seen = set()

    for frame_data in tracker.track_video(args.video, save_path=args.out, show=args.show):
        num_frames += 1
        for fighter in frame_data["fighters"]:
            max_ids_seen.add(fighter["track_id"])

        if num_frames % 30 == 0:
            print(f"Frame {num_frames}: {len(frame_data['fighters'])} fighters tracked "
                  f"(IDs so far: {sorted(max_ids_seen)})")

    print(f"\nDone. Processed {num_frames} frames. Output saved to: {args.out}")
    print(f"Unique track IDs seen: {sorted(max_ids_seen)}")
    print("If you see MANY more than 2 IDs, tracking is unstable -- "
          "this is exactly what Phase 2 (glove detection + re-ID) will fix.")
