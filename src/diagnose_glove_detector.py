"""
Quick diagnostic script: run the glove detector on ONE frame from a
video, at a very low confidence threshold, and save the result as an
image so we can visually check what (if anything) it's detecting.

Use this whenever the glove detector seems to be missing everything --
it's much faster than re-running the full video pipeline.
"""

import cv2
from pathlib import Path
from ultralytics import YOLO
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose glove detector on a single frame")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=30, help="Which frame number to test")
    parser.add_argument("--model", default="models/glove_detector/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.05, help="Very low threshold to see any weak detections")
    parser.add_argument("--out", default="data/output/glove_diagnostic.jpg")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read frame {args.frame} from {args.video}")
        exit(1)

    model = YOLO(args.model)
    result = model(frame, conf=args.conf, verbose=False)[0]

    print(f"\nTested frame {args.frame} at confidence threshold {args.conf}")
    print(f"Detections found: {len(result.boxes) if result.boxes is not None else 0}")

    if result.boxes is not None and len(result.boxes) > 0:
        confs = result.boxes.conf.cpu().numpy()
        print(f"Confidence scores: {sorted(confs, reverse=True)}")
    else:
        print("NO detections at all, even at conf=0.05 -- strong sign of a domain gap "
              "between training images and this real footage.")

    annotated = result.plot()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, annotated)
    print(f"\nSaved annotated frame to: {args.out}")
    print("Open that image to see any detected boxes (even faint/wrong ones).")
