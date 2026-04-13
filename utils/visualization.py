import cv2
import numpy as np
from typing import Dict, List
from collections import deque


RISK_COLORS = {
    "High": (0, 0, 255),
    "Medium": (0, 200, 255),
    "Low": (255, 170, 0),
}

# NEW: Scenario-specific accent colors for incident labels
SCENARIO_COLORS = {
    "rear_end": (0, 100, 255),
    "merging": (0, 200, 200),
    "crossing": (200, 0, 200),
    "head_on": (0, 0, 255),
    "hard_brake": (0, 200, 255),
    "swerve": (255, 100, 0),
}


def build_risk_map(incidents: List[dict]) -> Dict[int, str]:
    pr = {"High": 3, "Medium": 2, "Low": 1}
    out = {}
    for inc in incidents:
        risk = inc.get("risk", "Low")
        for k in ("actor_1", "actor_2"):
            actor = inc.get(k)
            if actor and "ID:" in actor:
                try:
                    tid = int(actor.split("ID:")[1].rstrip(")"))
                    if pr[risk] > pr.get(out.get(tid, ""), 0):
                        out[tid] = risk
                except Exception:
                    pass
    return out


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
    total_incidents: int
):
    risk_map = build_risk_map(incidents)

    for tid_raw, centroid, box, score, cls_id in tracks:
        tid = int(tid_raw)
        x1, y1, x2, y2 = box.astype(np.int32)
        cls_name = class_mapping.get(int(cls_id), "vehicle")
        risk = risk_map.get(tid)
        color = RISK_COLORS.get(risk, (200, 200, 200))
        thick = 3 if risk else 2

        label = f"ID:{tid} {cls_name} {int(score)}%"
        if risk:
            label += f" [{risk}]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - bl - 4)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for tid, hist in track_history.items():
        if len(hist) < 2:
            continue
        color = RISK_COLORS.get(risk_map.get(tid), (255, 170, 0))
        pts = np.array(hist, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2)

    y = 35
    for inc in incidents[:6]:
        risk = inc.get("risk", "Low")
        scenario = inc.get("scenario", "")
        color = SCENARIO_COLORS.get(scenario, RISK_COLORS.get(risk, (255, 255, 255)))

        # NEW: show scenario type in the overlay
        txt = f"[{risk}] {inc['type']}"
        if inc.get("ttc_s") is not None:
            txt += f" TTC:{inc['ttc_s']}s"
        if inc.get("distance_m") is not None:
            txt += f" D:{inc['distance_m']}m"
        if inc.get("lateral_acc_mps2") is not None:
            txt += f" LatAcc:{inc['lateral_acc_mps2']}m/s²"

        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (15, y - th - 5), (25 + tw, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += th + 14

    h = frame.shape[0]
    hud = f"Frame:{frame_idx}/{total_frames} FPS:{cur_fps:.1f} Avg:{avg_fps:.1f} Inc:{total_incidents}"
    cv2.putText(frame, hud, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    return frame