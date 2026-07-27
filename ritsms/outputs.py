"""§6-7 — Protocol data outputs.

  * TrajectoryWriter — per-frame per-road-user file (schema §7.1).
  * ConflictWriter   — per-pair conflict file (schema §7.2).
  * Reporter         — summary JSON, self-contained dashboard, BEV conflict
                       heatmap (cv2, no external deps), validation pack.

Metric columns (LocalX_M/Y_M, ConflictX_M/Y_M) are written only when the site
is metric-validated; otherwise null, with calibration_confidence recorded in
the report header (BEV-only baseline, §4.2).
"""

import os
import csv
import json
import base64
import logging
import numpy as np
import cv2
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

TRAJ_FIELDS = [
    "FrameNumber", "ObjectID", "ObjectType", "BB_X", "BB_Y", "BB_W", "BB_H",
    "X", "Y", "BEV_X", "BEV_Y", "LocalX_M", "LocalY_M", "TimeStamp",
    "LocalX_M_Smooth", "LocalY_M_Smooth", "Speed", "Acceleration", "Bearing",
    "Direction", "Length", "Width", "CorrectTimeStamp",
]

CONFLICT_FIELDS = [
    "Site_ID", "Conflict_ID", "RoadUser1_id", "RoadUser2_id",
    "RoadUser1_type", "RoadUser2_type", "Encounter_type", "ConflictLevel",
    "ttc", "pet", "drac", "deltav", "NFb3", "ConflictX_BEV", "ConflictY_BEV",
    "ConflictX_M", "ConflictY_M", "CaptureTimeStamp", "CorrectTimeStamp",
    "InteractionAngle",
]


class TrajectoryWriter:
    def __init__(self, path, metric_confident):
        self.path = path
        self.metric = metric_confident
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=TRAJ_FIELDS)
        self._w.writeheader()

    def write_frame(self, frame_idx, capture_ts_ms, correct_ts, tracks):
        for tid, trk in tracks.items():
            if trk._mean is None:
                continue
            bx = trk.last_bbox or [0, 0, 0, 0]
            img = trk.last_img_point or (0, 0)
            pos = trk.position_m
            m = self.metric
            self._w.writerow({
                "FrameNumber": frame_idx, "ObjectID": tid, "ObjectType": trk.cls_name,
                "BB_X": bx[0], "BB_Y": bx[1], "BB_W": bx[2] - bx[0], "BB_H": bx[3] - bx[1],
                "X": img[0], "Y": img[1],
                "BEV_X": round(float(pos[0]), 3), "BEV_Y": round(float(pos[1]), 3),
                "LocalX_M": round(float(pos[0]), 3) if m else "",
                "LocalY_M": round(float(pos[1]), 3) if m else "",
                "TimeStamp": round(capture_ts_ms, 1),
                "LocalX_M_Smooth": round(float(pos[0]), 3) if m else "",
                "LocalY_M_Smooth": round(float(pos[1]), 3) if m else "",
                "Speed": round(float(trk.speed), 3),
                "Acceleration": round(float(trk.acc), 3),
                "Bearing": round(float(trk.heading_deg), 1),
                "Direction": "",
                "Length": round(float(trk.length_m), 2), "Width": round(float(trk.width_m), 2),
                "CorrectTimeStamp": correct_ts if correct_ts is not None else "",
            })

    def close(self):
        self._f.close()


class ConflictWriter:
    def __init__(self, path, metric_confident):
        self.path = path
        self.metric = metric_confident
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=CONFLICT_FIELDS)
        self._w.writeheader()

    def write(self, e):
        m = self.metric
        self._w.writerow({
            "Site_ID": e["site_id"], "Conflict_ID": e["conflict_id"],
            "RoadUser1_id": e["roaduser1_id"], "RoadUser2_id": e["roaduser2_id"],
            "RoadUser1_type": e["roaduser1_type"], "RoadUser2_type": e["roaduser2_type"],
            "Encounter_type": e["encounter_type"], "ConflictLevel": e["level"],
            "ttc": e["ttc_s"] if e["ttc_s"] is not None else "",
            "pet": e["pet_s"] if e["pet_s"] is not None else "",
            "drac": e["drac_mps2"] if e["drac_mps2"] is not None else "",
            "deltav": e["deltav_mps"] if e["deltav_mps"] is not None else "",
            "NFb3": e["nfb3"],
            "ConflictX_BEV": e["conflict_x_bev"], "ConflictY_BEV": e["conflict_y_bev"],
            "ConflictX_M": e["conflict_x_bev"] if m else "",
            "ConflictY_M": e["conflict_y_bev"] if m else "",
            "CaptureTimeStamp": e.get("timestamp", ""),
            "CorrectTimeStamp": e.get("correct_timestamp") or "",
            "InteractionAngle": e["interaction_angle_deg"],
        })

    def close(self):
        self._f.close()


class Reporter:
    def __init__(self, cfg, site, fps):
        self.cfg = cfg
        self.site = site
        self.fps = fps
        self.events = []

    def add(self, incidents):
        self.events.extend(incidents)

    def summary(self):
        by_level = defaultdict(int)
        by_enc = defaultdict(int)
        by_gov = defaultdict(int)
        temporal = defaultdict(int)
        classes = defaultdict(int)
        win = max(int(self.fps * 10), 1)
        for e in self.events:
            by_level[e["level"]] += 1
            by_enc[e["encounter_type"]] += 1
            by_gov[e["governing_measure"]] += 1
            w = (e["frame"] // win) * 10
            temporal[f"{w}-{w+10}s"] += 1
            classes[e["roaduser1_type"]] += 1
            classes[e["roaduser2_type"]] += 1
        return {
            "total_conflicts": len(self.events),
            "level_distribution": dict(by_level),
            "encounter_distribution": dict(by_enc),
            "governing_measure": dict(by_gov),
            "temporal_analysis": dict(temporal),
            "involved_classes": dict(classes),
        }

    def heatmap_png(self, path, size=720):
        pts = [(e["conflict_x_bev"], e["conflict_y_bev"]) for e in self.events]
        canvas = np.zeros((size, size), dtype=np.float32)
        if pts:
            xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
            x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
            sx = (size - 40) / max(x1 - x0, 1e-3); sy = (size - 40) / max(y1 - y0, 1e-3)
            s = min(sx, sy)
            for x, y in pts:
                px = int(20 + (x - x0) * s); py = int(20 + (y - y0) * s)
                if 0 <= px < size and 0 <= py < size:
                    cv2.circle(canvas, (px, py), 8, 1.0, -1)
            canvas = cv2.GaussianBlur(canvas, (0, 0), 12)
            if canvas.max() > 0:
                canvas = canvas / canvas.max()
        img = cv2.applyColorMap((canvas * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(path, img)
        return path

    def save(self, report_path, dashboard_path, heatmap_path, tracking_quality):
        s = self.summary()
        self.heatmap_png(heatmap_path)
        payload = {
            "generated_at": datetime.now().isoformat(),
            "site_id": self.cfg.site_id,
            "calibration_confidence": self.site.calibration_confidence,
            "metric_outputs_valid": self.site.metric_confident,
            "reprojection_rmse_m": self.site.reprojection_rmse_m,
            "reprojection_p95_m": self.site.reprojection_p95_m,
            "roi_enabled": self.site.roi_enabled,
            "summary": s,
            "tracking_quality": tracking_quality,
            "conflicts": self.events,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._dashboard(dashboard_path, s, payload, heatmap_path)
        return s

    def _dashboard(self, path, s, payload, heatmap_path):
        try:
            with open(heatmap_path, "rb") as f:
                hb = base64.b64encode(f.read()).decode("ascii")
        except Exception:
            hb = ""
        lv = s["level_distribution"]
        rows_enc = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in
                           sorted(s["encounter_distribution"].items(), key=lambda x: -x[1]))
        tq = payload["tracking_quality"] or {}
        rows_tq = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in tq.items())
        conf = payload["calibration_confidence"]
        html = f"""<!doctype html><html><head><meta charset=utf-8><title>RITSMS Conflict Report</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;background:#12141c;color:#e7e9f0;margin:0;padding:24px}}
h1{{color:#4c8bf5}} .cards{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#1b1e2b;border-radius:10px;padding:16px 22px;min-width:120px;text-align:center}}
.card .n{{font-size:2em;font-weight:700}} .card .l{{color:#8b90a3;font-size:.85em}}
table{{border-collapse:collapse;margin:8px 0;background:#1b1e2b;border-radius:8px;overflow:hidden}}
td,th{{border-bottom:1px solid #2a2e40;padding:6px 14px;text-align:left}}
.flex{{display:flex;gap:24px;flex-wrap:wrap}} img{{max-width:520px;border-radius:8px}}
.badge{{padding:3px 10px;border-radius:20px;background:#2a2e40;font-size:.85em}}</style></head><body>
<h1>🚦 RITSMS Near-Miss 2.0 — Conflict Report</h1>
<p>Site <b>{payload['site_id']}</b> · calibration: <span class=badge>{conf}</span>
 · metric outputs valid: <b>{payload['metric_outputs_valid']}</b> · ROI: <b>{payload['roi_enabled']}</b></p>
<div class=cards>
  <div class=card><div class=n>{s['total_conflicts']}</div><div class=l>Conflicts</div></div>
  <div class=card><div class=n style="color:#ff4444">{lv.get('CRITICAL',0)}</div><div class=l>Critical</div></div>
  <div class=card><div class=n style="color:#ff8c00">{lv.get('WARNING',0)}</div><div class=l>Warning</div></div>
  <div class=card><div class=n>{s['governing_measure'].get('TTC',0)}</div><div class=l>TTC</div></div>
  <div class=card><div class=n>{s['governing_measure'].get('PET',0)}</div><div class=l>PET</div></div>
</div>
<div class=flex>
  <div><h3>Conflict heatmap (BEV)</h3><img src="data:image/png;base64,{hb}"></div>
  <div><h3>Encounter types</h3><table><tr><th>Type</th><th>#</th></tr>{rows_enc}</table></div>
  <div><h3>Tracking quality (AC2)</h3><table><tr><th>Metric</th><th>Value</th></tr>{rows_tq}</table></div>
</div>
<p style="color:#8b90a3">Generated {payload['generated_at']}. TTC/PET are acceptance-grade in BEV-only mode;
DRAC/Delta-V/metric coordinates are conditional until the site is metric-validated.</p>
</body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
