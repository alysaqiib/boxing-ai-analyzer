"""
Robustness test: simulated camera cut
------------------------------------------
This script takes REAL tracked pose data from your video, then
artificially injects a sudden position jump partway through -- exactly
what a hard camera cut (a sudden switch to a different angle) would
look like to the tracker.

It then compares strike detection with vs. without that injected jump,
to see concretely whether a camera cut would cause a FALSE strike to
be detected right at the cut point.

This does NOT modify any of your actual pipeline files -- it's a
standalone diagnostic.
"""

import copy
import numpy as np

from glove_confirmed_tracker import run_glove_confirmed_tracking
from reid_fighter_tracker import compute_appearance_signatures, cluster_into_two_identities
from strike_classifier import build_identity_frame_series, detect_strikes_for_identity


def inject_fake_cut(frame_series: dict, cut_at_frame: int, jump_pixels: float = 400.0) -> dict:
    """Returns a COPY of frame_series where every keypoint after cut_at_frame
    is shifted by a large offset -- simulating the sudden jump a real camera
    cut would cause."""
    modified = copy.deepcopy(frame_series)
    for fi, fighter_entry in modified.items():
        if fi >= cut_at_frame:
            kpts = fighter_entry["keypoints"]
            kpts[:, 0] += jump_pixels  # shift all x-coordinates
            bbox = fighter_entry["bbox"]
            fighter_entry["bbox"] = [bbox[0] + jump_pixels, bbox[1], bbox[2] + jump_pixels, bbox[3]]
    return modified


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test: does a simulated camera cut cause false strikes?")
    parser.add_argument("--video", required=True)
    parser.add_argument("--pose_model", default="yolov8n-pose.pt")
    parser.add_argument("--glove_model", default="models/glove_detector/weights/best.pt")
    parser.add_argument("--cut_at_frame", type=int, default=300,
                         help="Which frame to inject the fake cut at (pick a frame in the middle of your clip)")
    args = parser.parse_args()

    print("Running real pipeline (pose + glove + re-ID) on your video...")
    history, per_id_stats = run_glove_confirmed_tracking(
        args.video, pose_model_path=args.pose_model, glove_model_path=args.glove_model
    )
    signatures = compute_appearance_signatures(args.video, history)
    track_id_to_identity, identity_stats = cluster_into_two_identities(per_id_stats, signatures)
    frame_series_by_identity = build_identity_frame_series(history, track_id_to_identity)

    print(f"\nInjecting a simulated camera cut at frame {args.cut_at_frame}...\n")

    for identity, frame_series in frame_series_by_identity.items():
        original_strikes = detect_strikes_for_identity(frame_series)

        cut_series = inject_fake_cut(frame_series, args.cut_at_frame)
        cut_strikes = detect_strikes_for_identity(cut_series)

        original_frames = {s["frame_idx"] for s in original_strikes}
        cut_frames = {s["frame_idx"] for s in cut_strikes}
        new_false_strikes = cut_frames - original_frames

        print(f"Fighter {identity}:")
        print(f"  Strikes WITHOUT simulated cut: {len(original_strikes)}")
        print(f"  Strikes WITH simulated cut:    {len(cut_strikes)}")
        if new_false_strikes:
            near_cut = [f for f in new_false_strikes if abs(f - args.cut_at_frame) < 10]
            print(f"  NEW false strike(s) appeared: {sorted(new_false_strikes)}")
            if near_cut:
                print(f"  -> {len(near_cut)} of these are suspiciously close to the cut point "
                      f"(frame {args.cut_at_frame}) -- this confirms the vulnerability.")
        else:
            print("  No new false strikes appeared at the cut point.")
        print()

    print("CONCLUSION: if you saw 'NEW false strike(s)' near the cut point above, "
          "that confirms camera cuts can cause false strike detections in the current pipeline. "
          "A fix would involve detecting large frame-to-frame position jumps and skipping "
          "strike detection for a few frames around them.")
