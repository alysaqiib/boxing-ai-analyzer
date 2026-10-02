"""
Live side analytics panel
------------------------------
Composites a broadcast-style side panel onto the video, showing three
live-updating visualizations as the fight plays:

  1. HIT-ZONE DIAGRAM -- a simple body outline per fighter, with markers
     accumulating at head/body/leg as strikes land on that zone.
  2. POSITION MAP -- a simplified 2D map showing roughly where each
     fighter is positioned.
  3. ACTIVITY GRAPH -- a scrolling line chart of cumulative strikes
     thrown over time, per fighter.

HONEST LIMITATIONS
  - Position map is NOT a true bird's-eye/top-down view -- we only have
    one side-angle camera with no real depth information. It's a
    simplified 2D proxy based on horizontal position in frame (and a
    rough vertical proxy from bounding-box size), not an accurate
    representation of ring position. Treat it as stylized, not precise.
  - Hit-zone diagram uses the SAME "Landed" classifications from
    strike_resolution.py, so it inherits all of that module's accuracy
    limitations too.
"""

import cv2
import numpy as np
from collections import defaultdict

PANEL_WIDTH = 340
BG_COLOR = (22, 20, 18)
GRID_COLOR = (50, 48, 45)
TEXT_COLOR = (220, 220, 220)
MUTED_COLOR = (130, 130, 130)


class LivePanelState:
    """Holds running/accumulating data needed to draw the panel, updated frame by frame."""

    def __init__(self, strikes_by_identity: dict, total_frames: int):
        self.hit_zone_counts = defaultdict(lambda: defaultdict(int))  # identity -> {zone: count}
        self.activity_totals = defaultdict(int)  # identity -> cumulative strike count so far
        self.activity_history = defaultdict(list)  # identity -> [cumulative count per rendered frame]

        # Index strikes by the frame they occur on, per identity, for quick lookup during rendering
        self.strikes_by_frame = defaultdict(list)  # frame_idx -> [(identity, strike), ...]
        for identity, strikes in strikes_by_identity.items():
            for s in strikes:
                self.strikes_by_frame[s["frame_idx"]].append((identity, s))

        self.total_frames = max(total_frames, 1)

    def update(self, frame_idx: int):
        """Call once per rendered frame, in order, to advance the accumulators."""
        for identity, strike in self.strikes_by_frame.get(frame_idx, []):
            self.activity_totals[identity] += 1
            if strike.get("result") == "Landed" and strike.get("target_zone"):
                self.hit_zone_counts[identity][strike["target_zone"]] += 1

        for identity in ["A", "B"]:
            self.activity_history[identity].append(self.activity_totals[identity])


def _draw_body_outline(panel, x, y, w, h, color):
    """A very simple stick-figure-ish body outline: head circle, torso line, leg lines."""
    head_r = int(w * 0.18)
    head_cx = x + w // 2
    head_cy = y + head_r + 2
    cv2.circle(panel, (head_cx, head_cy), head_r, color, 1, cv2.LINE_AA)

    torso_top = (head_cx, head_cy + head_r)
    torso_bottom = (head_cx, y + int(h * 0.62))
    cv2.line(panel, torso_top, torso_bottom, color, 1, cv2.LINE_AA)

    leg_spread = int(w * 0.16)
    leg_bottom_y = y + h
    cv2.line(panel, torso_bottom, (head_cx - leg_spread, leg_bottom_y), color, 1, cv2.LINE_AA)
    cv2.line(panel, torso_bottom, (head_cx + leg_spread, leg_bottom_y), color, 1, cv2.LINE_AA)

    zone_points = {
        "Head": (head_cx, head_cy),
        "Body": (head_cx, (torso_top[1] + torso_bottom[1]) // 2),
        "Leg": (head_cx, int(leg_bottom_y - h * 0.12)),
    }
    return zone_points


def draw_hitzone_section(panel, x, y, w, h, identities_data: dict, identity_colors: dict):
    """identities_data: {"A": {"Head": n, "Body": n, "Leg": n}, "B": {...}}"""
    cv2.putText(panel, "HIT ZONES", (x, y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    diagram_w = w // 2 - 15
    diagram_h = h - 40
    diagram_y = y + 30

    for i, identity in enumerate(["A", "B"]):
        dx = x + i * (diagram_w + 30)
        color = identity_colors.get(identity, (200, 200, 200))
        zone_points = _draw_body_outline(panel, dx, diagram_y, diagram_w, diagram_h, MUTED_COLOR)

        counts = identities_data.get(identity, {})
        max_count = max(counts.values()) if counts else 0
        for zone, (zx, zy) in zone_points.items():
            count = counts.get(zone, 0)
            if count > 0:
                radius = min(4 + int(count * 1.5), 14)
                cv2.circle(panel, (zx, zy), radius, color, -1, cv2.LINE_AA)
                cv2.putText(panel, str(count), (zx + radius + 3, zy + 4),
                            cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)

        cv2.putText(panel, f"Fighter {identity}", (dx, diagram_y + diagram_h + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_position_map_section(panel, x, y, w, h, positions: dict, identity_colors: dict):
    """positions: {"A": (norm_x, norm_y) in 0-1 range, "B": (...)}.
    NOTE: this is a stylized 2D proxy, not a true bird's-eye view (see module docstring)."""
    cv2.putText(panel, "POSITION (approx.)", (x, y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    map_x, map_y = x, y + 28
    map_w, map_h = w, h - 40
    cv2.rectangle(panel, (map_x, map_y), (map_x + map_w, map_y + map_h), GRID_COLOR, 1, cv2.LINE_AA)
    # Simple ring-rope suggestion lines
    for frac in [0.33, 0.66]:
        yy = int(map_y + map_h * frac)
        cv2.line(panel, (map_x, yy), (map_x + map_w, yy), GRID_COLOR, 1, cv2.LINE_AA)

    for identity, (nx, ny) in positions.items():
        color = identity_colors.get(identity, (200, 200, 200))
        px = int(map_x + np.clip(nx, 0, 1) * map_w)
        py = int(map_y + np.clip(ny, 0, 1) * map_h)
        cv2.circle(panel, (px, py), 7, color, -1, cv2.LINE_AA)
        cv2.putText(panel, identity, (px - 4, py + 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


def draw_activity_graph_section(panel, x, y, w, h, activity_history: dict, identity_colors: dict):
    cv2.putText(panel, "ACTIVITY (cumulative strikes)", (x, y + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    graph_x, graph_y = x, y + 28
    graph_w, graph_h = w, h - 40
    cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), GRID_COLOR, 1, cv2.LINE_AA)

    max_val = 1
    for identity, history in activity_history.items():
        if history:
            max_val = max(max_val, history[-1])

    for identity, history in activity_history.items():
        if len(history) < 2:
            continue
        color = identity_colors.get(identity, (200, 200, 200))
        points = []
        for i, val in enumerate(history):
            px = int(graph_x + (i / max(len(history) - 1, 1)) * graph_w)
            py = int(graph_y + graph_h - (val / max_val) * (graph_h - 8) - 4)
            points.append((px, py))
        for i in range(1, len(points)):
            cv2.line(panel, points[i - 1], points[i], color, 2, cv2.LINE_AA)

        if history:
            label_y = graph_y + 16 + 16 * list(activity_history.keys()).index(identity)
            cv2.putText(panel, f"{identity}: {history[-1]}", (graph_x + 8, label_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, cv2.LINE_AA)


def render_frame_with_panel(frame: np.ndarray, frame_idx: int, panel_state: LivePanelState,
                             positions: dict, identity_colors: dict) -> np.ndarray:
    """Composites the full side panel onto one frame. Call once per frame, in order,
    AFTER calling panel_state.update(frame_idx)."""
    h, w = frame.shape[:2]
    panel = np.full((h, PANEL_WIDTH, 3), BG_COLOR, dtype=np.uint8)

    section_h = h // 3
    draw_hitzone_section(panel, 15, 10, PANEL_WIDTH - 30, section_h - 20, panel_state.hit_zone_counts, identity_colors)
    draw_position_map_section(panel, 15, section_h + 10, PANEL_WIDTH - 30, section_h - 20, positions, identity_colors)
    draw_activity_graph_section(panel, 15, 2 * section_h + 10, PANEL_WIDTH - 30, section_h - 20,
                                 panel_state.activity_history, identity_colors)

    return np.hstack([frame, panel])


def estimate_positions_from_frame(frame_data: dict, track_id_to_identity: dict, frame_width: int, frame_height: int) -> dict:
    """Rough 2D position proxy: horizontal = bbox center x / frame width.
    Vertical = a crude proxy using bbox height (taller/closer bbox -> lower
    on screen = further forward in a typical boxing-ring camera angle).
    This is NOT a real depth/position measurement -- see module docstring."""
    positions = {}
    for fighter in frame_data["fighters"]:
        identity = track_id_to_identity.get(fighter["track_id"])
        if identity is None:
            continue
        x1, y1, x2, y2 = fighter["bbox"]
        cx = (x1 + x2) / 2
        norm_x = cx / frame_width
        bbox_h = (y2 - y1) / frame_height
        norm_y = float(np.clip(1.0 - bbox_h, 0.05, 0.95))  # taller bbox (closer) -> higher on map
        positions[identity] = (norm_x, norm_y)
    return positions
