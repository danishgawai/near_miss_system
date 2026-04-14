"""
Visualization helpers for the near-miss detection pipeline.

Fixes applied:
  - Overlay text and bounding boxes now use the same RISK_COLORS palette
    so colours always match.
  - Overlay reads edge_distance_m (updated field name) with fallback to
    centre_distance_m so distance is always shown when available.
  - Trajectory polylines use the same risk colour as the bounding box.
  - risk_index is shown in the overlay when present.
"""

import cv2
import numpy as np
from typing import Dict, List
from collections import deque


# Shared palette — used by BOTH bounding boxes AND overlay text
RISK_COLORS = {
    "High":   (0,   0,   255),   # red
    "Medium": (0,   165, 255),   # orange
    "Low":    (0,   200, 80),    # green
}

# Fallback when no risk level is assigned (neutral track)
_DEFAULT_COLOR = (180, 180, 180)

# Scenario accent — only used for single-object events (hard_brake, swerve)
# which have no risk-level colour of their own
SCENARIO_COLORS = {
    "hard_brake": (0,   200, 255),   # cyan
    "swerve":     (255, 100, 0),     # blue-orange
}


def build_risk_map(incidents: List[dict]) -> Dict[int, str]:
    """Return a dict of track_id → highest risk label for the current frame."""
    priority = {"High": 3, "Medium": 2, "Low": 1}
    out: Dict[int, str] = {}
    for inc in incidents:
        risk = inc.get("risk", "Low")
        for key in ("actor_1", "actor_2"):
            actor = inc.get(key)
            if actor and "ID:" in actor:
                try:
                    tid = int(actor.split("ID:")[1].rstrip(")"))
                    if priority.get(risk, 0) > priority.get(out.get(tid, ""), 0):
                        out[tid] = risk
                except Exception:
                    pass
    return out


def _incident_color(inc: dict) -> tuple:
    """
    Return the BGR colour to use for an incident overlay line.
    Always derived from RISK_COLORS so it matches the bounding box colour.
    Falls back to SCENARIO_COLORS for single-object events that carry no RI.
    """
    risk = inc.get("risk")
    if risk in RISK_COLORS:
        return RISK_COLORS[risk]
    scenario = inc.get("scenario", "")
    return SCENARIO_COLORS.get(scenario, (255, 255, 255))


def draw_frame(
    frame,
    tracks,
    class_mapping: dict,
    track_history: Dict[int, deque],
    incidents: List[dict],
    frame_idx: int,
    total_frames: int,
    cur_fps: float,
    avg_fps: float,
    total_incidents: int,
):
    risk_map = build_risk_map(incidents)

    # ------------------------------------------------------------------ #
    # 1. Bounding boxes + per-track labels                                #
    # ------------------------------------------------------------------ #
    for tid_raw, centroid, box, score, cls_id in tracks:
        tid      = int(tid_raw)
        x1, y1, x2, y2 = box.astype(np.int32)
        cls_name = class_mapping.get(int(cls_id), "vehicle")
        risk     = risk_map.get(tid)
        color    = RISK_COLORS.get(risk, _DEFAULT_COLOR)
        thick    = 3 if risk else 2

        label = f"ID:{tid} {cls_name} {int(score)}%"
        if risk:
            label += f" [{risk}]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg_y1 = max(0, y1 - th - bl - 4)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame, label,
            (x1 + 3, y1 - bl - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------ #
    # 2. Trajectory polylines — same colour as the bounding box           #
    # ------------------------------------------------------------------ #
    for tid, hist in track_history.items():
        if len(hist) < 2:
            continue
        color = RISK_COLORS.get(risk_map.get(tid), _DEFAULT_COLOR)
        pts   = np.array(hist, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # 3. Incident overlay (top-left, max 6 lines)                         #
    #    Colour matches the bounding box via _incident_color()            #
    # ------------------------------------------------------------------ #
    y = 35
    for inc in incidents[:6]:
        color = _incident_color(inc)          # ← always from RISK_COLORS

        # Build label
        txt = f"[{inc.get('risk','?')}] {inc['type']}"

        ri = inc.get("risk_index")
        if ri is not None:
            txt += f"  RI:{ri:.2f}"

        if inc.get("ttc_s") is not None:
            txt += f"  TTC:{inc['ttc_s']}s"

        # edge_distance_m is the new field; fall back to centre_distance_m
        dist = inc.get("edge_distance_m") or inc.get("centre_distance_m")
        if dist is not None:
            txt += f"  D:{dist}m"

        if inc.get("lateral_acc_mps2") is not None:
            txt += f"  LatAcc:{inc['lateral_acc_mps2']}m/s²"

        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        cv2.rectangle(frame, (12, y - th - 5), (22 + tw, y + 5), (0, 0, 0), -1)
        cv2.putText(
            frame, txt,
            (17, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA,
        )
        y += th + 14

    # ------------------------------------------------------------------ #
    # 4. HUD bar at the bottom                                            #
    # ------------------------------------------------------------------ #
    hud = (
        f"Frame:{frame_idx}/{total_frames}  "
        f"FPS:{cur_fps:.1f}  Avg:{avg_fps:.1f}  "
        f"Inc:{total_incidents}"
    )
    h = frame.shape[0]
    (tw, th), bl = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (0, h - th - bl - 8), (tw + 14, h), (0, 0, 0), -1)
    cv2.putText(
        frame, hud,
        (7, h - bl - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
    )

    return frame