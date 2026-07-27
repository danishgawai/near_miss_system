import json
from datetime import datetime
from collections import defaultdict

import csv


class TelemetryLogger:
    def __init__(self, path: str):
        self.path = path
        self.fp = None
        self.writer = None

    def open(self):
        self.fp = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.fp)
        self.writer.writerow([
            "frame",
            "timestamp",
            "track_id",
            "class_name",
            "bev_x",
            "bev_y",
            "speed_mps",
            "heading_deg",
            "acc_mps2",
            "risk_index",
        ])

    def log(self, frame_idx, timestamp, track_id, class_name, bev_x, bev_y, speed, heading, acc, risk_index):
        if self.writer is None:
            return
        self.writer.writerow([
            frame_idx,
            timestamp,
            track_id,
            class_name,
            round(float(bev_x), 4),
            round(float(bev_y), 4),
            round(float(speed), 4),
            round(float(heading), 4),
            round(float(acc), 4),
            round(float(risk_index), 4),
        ])

    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None
            self.writer = None

class Reporter:
    def __init__(self, fps: float, calibration_confidence: str = "bev_only"):
        self.fps = fps
        # BEV-only baseline: metric outputs (speed m/s, DRAC, Delta-V, *_M) are
        # conditional. Recorded in the report header so downstream review knows
        # which columns are acceptance-grade vs diagnostic.
        self.calibration_confidence = calibration_confidence
        self.events = []

    def add(self, incidents):
        self.events.extend(incidents)

    def summary(self):
        total = len(self.events)
        if total == 0:
            return {"total_near_misses": 0}

        level_dist = defaultdict(int)
        type_dist = defaultdict(int)
        scenario_dist = defaultdict(int)
        metric_dist = defaultdict(int)
        temporal = defaultdict(int)
        classes = defaultdict(int)
        frame_count = defaultdict(int)
        window = max(int(self.fps * 10), 1)

        for e in self.events:
            level_dist[e.get("level", "WARNING")] += 1
            type_dist[e["type"]] += 1
            scenario_dist[e.get("scenario", "unknown")] += 1
            metric_dist[e.get("metric_kind", "n/a")] += 1

            w = (e["frame"] // window) * 10
            temporal[f"{w}-{w+10}s"] += 1
            for k in ("actor_1", "actor_2"):
                a = e.get(k)
                if a:
                    classes[a.split(" (")[0]] += 1
            frame_count[e["frame"]] += 1

        peak = sorted(frame_count.items(), key=lambda x: -x[1])[:5]
        return {
            "total_near_misses": total,
            "level_distribution": dict(level_dist),
            "type_distribution": dict(type_dist),
            "scenario_distribution": dict(scenario_dist),
            "metric_distribution": dict(metric_dist),
            "temporal_analysis": dict(temporal),
            "involved_classes": dict(classes),
            "peak_frames": [{"frame": f, "count": c} for f, c in peak]
        }

    def save_json(self, path: str):
        s = self.summary()
        payload = {
            "generated_at": datetime.now().isoformat(),
            "calibration_confidence": self.calibration_confidence,
            "metric_outputs_valid": self.calibration_confidence == "metric_validated",
            "summary": s,
            "incidents": self.events
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return s


RISK_HEX = {
    "CRITICAL": "#ff4444",
    "WARNING": "#ff8c00",
    "SAFE": "#4caf50",
}


def save_dashboard(summary: dict, path: str):
    risk_dist = summary.get("level_distribution", {})
    risk_labels = json.dumps(list(risk_dist.keys()))
    risk_values = json.dumps(list(risk_dist.values()))
    # Colours mapped BY LABEL — a fixed array silently mismatches when the
    # label order varies (dict insertion order follows first occurrence).
    risk_colors = json.dumps([RISK_HEX.get(k, "#4caf50") for k in risk_dist.keys()])
    temp_labels = json.dumps(list(summary.get("temporal_analysis", {}).keys()))
    temp_values = json.dumps(list(summary.get("temporal_analysis", {}).values()))
    cls_labels = json.dumps(list(summary.get("involved_classes", {}).keys()))
    cls_values = json.dumps(list(summary.get("involved_classes", {}).values()))
    # NEW: scenario chart data
    scen_labels = json.dumps(list(summary.get("scenario_distribution", {}).keys()))
    scen_values = json.dumps(list(summary.get("scenario_distribution", {}).values()))

    total = summary.get("total_near_misses", 0)
    critical = risk_dist.get("CRITICAL", 0)
    warning = risk_dist.get("WARNING", 0)
    safe = risk_dist.get("SAFE", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Near-Miss Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
  h1 {{ text-align: center; color: #e94560; }}
  .stats {{ display: flex; justify-content: center; gap: 30px; margin: 20px 0; flex-wrap: wrap; }}
  .card {{ background: #16213e; border-radius: 12px; padding: 20px 30px; text-align: center; min-width: 140px; }}
  .card .num {{ font-size: 2.5em; font-weight: bold; }}
  .card .lbl {{ font-size: 0.9em; color: #aaa; margin-top: 5px; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 30px; }}
  .chart-box {{ background: #16213e; border-radius: 12px; padding: 20px; }}
  canvas {{ max-height: 300px; }}
</style>
</head>
<body>
<h1>🚨 Near-Miss Incident Dashboard</h1>
<div class="stats">
  <div class="card"><div class="num">{total}</div><div class="lbl">Total</div></div>
  <div class="card"><div class="num" style="color:#ff4444">{critical}</div><div class="lbl">Critical</div></div>
  <div class="card"><div class="num" style="color:#ff8c00">{warning}</div><div class="lbl">Warning</div></div>
  <div class="card"><div class="num" style="color:#4caf50">{safe}</div><div class="lbl">Safe</div></div>
</div>
<div class="charts">
  <div class="chart-box"><canvas id="c1"></canvas></div>
  <div class="chart-box"><canvas id="c2"></canvas></div>
  <div class="chart-box"><canvas id="c3"></canvas></div>
  <div class="chart-box"><canvas id="c4"></canvas></div>
</div>
<script>
new Chart(document.getElementById('c1'),{{
  type:'doughnut',
  data:{{labels:{risk_labels},datasets:[{{data:{risk_values},backgroundColor:{risk_colors}}}]}},
  options:{{plugins:{{title:{{display:true,text:'Risk Distribution',color:'#eee'}}}}}}
}});
new Chart(document.getElementById('c2'),{{
  type:'bar',
  data:{{labels:{temp_labels},datasets:[{{label:'Incidents',data:{temp_values},backgroundColor:'#e94560'}}]}},
  options:{{plugins:{{title:{{display:true,text:'Temporal Analysis',color:'#eee'}}}},scales:{{y:{{ticks:{{color:'#aaa'}}}},x:{{ticks:{{color:'#aaa'}}}}}}}}
}});
new Chart(document.getElementById('c3'),{{
  type:'bar',
  data:{{labels:{cls_labels},datasets:[{{label:'Count',data:{cls_values},backgroundColor:'#0f3460'}}]}},
  options:{{indexAxis:'y',plugins:{{title:{{display:true,text:'Object Classes Involved',color:'#eee'}}}},scales:{{x:{{ticks:{{color:'#aaa'}}}},y:{{ticks:{{color:'#aaa'}}}}}}}}
}});
new Chart(document.getElementById('c4'),{{
  type:'doughnut',
  data:{{labels:{scen_labels},datasets:[{{data:{scen_values},backgroundColor:['#0f3460','#e94560','#a855f7','#f59e0b','#10b981']}}]}},
  options:{{plugins:{{title:{{display:true,text:'Scenario Types',color:'#eee'}}}}}}
}});
</script>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)