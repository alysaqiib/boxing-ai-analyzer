"""
Phase 2: Train a custom glove detector
----------------------------------------
Fine-tunes a pretrained YOLOv8 object-detection model on the boxing
glove dataset (downloaded from Roboflow) so we can later use glove
detections to:
  1. Confirm "this tracked person is a real fighter" (filters out
     referees/cornermen who won't have gloves detected near them)
  2. Feed glove position into strike classification in Phase 3

This trains a DETECTION model (bounding boxes), separate from the
POSE model used in pose_tracker.py. We'll combine both later.
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


def train_glove_detector(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "models",
    name: str = "glove_detector",
    workers: int = 4,
):
    """
    data_yaml: path to the dataset's data.yaml (from Roboflow export)
    base_model: pretrained YOLOv8 weights to fine-tune from.
                'yolov8n.pt' = nano, fastest, good starting point.
                Swap to 'yolov8s.pt' for better accuracy once this works,
                your RTX 4050 can handle it.
    epochs: how many passes over the full dataset. 50 is a reasonable
            starting point for ~1500 images; increase if the model is
            still improving at the end (check the loss curves).
    imgsz: training image resolution. 640 is the YOLO standard.
    batch: images per training step. 16 is safe for a 6GB GPU; lower
           to 8 if you hit an out-of-memory error.
    device: '0' = first GPU. Use 'cpu' only as a last resort (much slower).
    project/name: results save to {project}/{name}/ (e.g. models/glove_detector/)
    """
    model = YOLO(base_model)  # loads/auto-downloads pretrained weights

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=15,  # stop early if val loss stops improving for 15 epochs
        plots=True,   # saves loss/mAP curves as images -- check these after training
        workers=workers,  # dataloader worker processes; set to 0 if training hangs
                           # (known WSL2 issue: limited /dev/shm can deadlock workers)
    )

    print("\nTraining complete.")
    print(f"Best weights saved to: {project}/{name}/weights/best.pt")
    print("Use that path with pose_tracker.py's glove-filtering step in Phase 2b.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Train glove detector")
    parser.add_argument("--data", required=True, help="Path to dataset's data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base pretrained model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--workers", type=int, default=4,
                         help="Dataloader worker processes. Set to 0 if training hangs/freezes (common WSL2 issue).")
    args = parser.parse_args()

    train_glove_detector(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )
