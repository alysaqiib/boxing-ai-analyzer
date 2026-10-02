import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from strike_classifier import detect_strikes_for_identity, _wrist_speed_series
from run_pipeline import filter_unresolved_strikes


def _fighter_frame(wrist_x, wrist_y=100.0, elbow_x=120.0):
    keypoints = np.zeros((17, 3), dtype=float)
    keypoints[:, 2] = 0.9
    keypoints[5] = [100.0, 100.0, 0.9]   # left shoulder
    keypoints[7] = [elbow_x, 100.0, 0.9]   # left elbow
    keypoints[9] = [wrist_x, wrist_y, 0.9]  # left wrist
    return {"keypoints": keypoints, "bbox": [0.0, 0.0, 200.0, 200.0]}


class StrikeClassifierTests(unittest.TestCase):
    def test_small_wrist_jitter_is_not_a_strike(self):
        series = {
            frame: _fighter_frame(145.0 + (frame % 2) * 2.0)
            for frame in range(10)
        }
        self.assertEqual(detect_strikes_for_identity(series, speed_threshold=0.30), [])

    def test_clear_arm_extension_is_detected(self):
        series = {}
        for frame in range(10):
            wrist_x = 145.0 if frame < 4 else 255.0 if frame == 4 else 260.0
            elbow_x = 120.0 if frame < 4 else 150.0 if frame == 4 else 152.0
            series[frame] = _fighter_frame(wrist_x, elbow_x=elbow_x)
        strikes = detect_strikes_for_identity(series, speed_threshold=0.30)
        self.assertEqual(len(strikes), 1)
        self.assertEqual(strikes[0]["side"], "left")

    def test_unresolved_events_are_not_user_facing_strikes(self):
        strikes = {
            "A": [
                {"frame_idx": 1, "result": "Unknown"},
                {"frame_idx": 2, "result": "Missed"},
            ]
        }
        filtered = filter_unresolved_strikes(strikes)
        self.assertEqual([item["result"] for item in filtered["A"]], ["Missed"])

    def test_wrist_speed_is_fps_invariant(self):
        series_30 = {
            frame: _fighter_frame(145.0 + frame * 3.0)
            for frame in range(4)
        }
        series_60 = {
            frame: _fighter_frame(145.0 + frame * 1.5)
            for frame in range(4)
        }
        speed_30 = _wrist_speed_series(series_30, 9, fps=30)[0][1]
        speed_60 = _wrist_speed_series(series_60, 9, fps=60)[0][1]
        self.assertAlmostEqual(speed_30, speed_60, places=6)

    def test_adjacent_opposite_hand_events_are_not_two_punches(self):
        series = {}
        for frame in range(12):
            left_x = 145.0 if frame < 4 else 255.0 if frame == 4 else 260.0
            right_x = 145.0 if frame < 5 else 255.0 if frame == 5 else 260.0
            entry = _fighter_frame(left_x, elbow_x=150.0 if frame >= 4 else 120.0)
            entry["keypoints"][6] = [100.0, 100.0, 0.9]
            entry["keypoints"][8] = [120.0, 105.0, 0.9]
            entry["keypoints"][10] = [right_x, 105.0, 0.9]
            series[frame] = entry
        strikes = detect_strikes_for_identity(series, speed_threshold=0.30)
        self.assertLessEqual(len(strikes), 1)

    def test_whole_body_translation_is_not_arm_extension(self):
        series = {}
        for frame in range(10):
            shift = frame * 20.0
            entry = _fighter_frame(145.0 + shift, elbow_x=120.0 + shift)
            entry["keypoints"][5] = [100.0 + shift, 100.0, 0.9]
            series[frame] = entry
        self.assertEqual(detect_strikes_for_identity(series, speed_threshold=0.30), [])


if __name__ == "__main__":
    unittest.main()
