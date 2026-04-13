import math
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List
from utils.motion import TrackState


class NearMissEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pair_persist = defaultdict(int)
        self.pair_cooldown = defaultdict(int)

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _ttc_cv(p1, v1, p2, v2, tmax: float):
        rel_p = p2 - p1
        rel_v = v2 - v1
        v2n = float(np.dot(rel_v, rel_v))
        dnow = float(np.linalg.norm(rel_p))
        if dnow < 1e-6:
            return 0.0, 0.0
        if v2n < 1e-8:
            return float("inf"), dnow

        t_star = -float(np.dot(rel_p, rel_v)) / v2n
        if t_star < 0 or t_star > tmax:
            return float("inf"), dnow
        return t_star, dnow

    @staticmethod
    def _heading_conflict_deg(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return math.degrees(math.acos(c))

    def _risk(self, ttc, dist, speed_diff, cls1, cls2):
        if ttc <= self.cfg.ttc_high_s:
            p = 5
        elif ttc <= self.cfg.ttc_med_s:
            p = 4
        elif ttc <= self.cfg.ttc_low_s:
            p = 3
        elif dist <= self.cfg.proximity_gate_m * 0.5:
            p = 2
        else:
            p = 1

        diff_kmh = speed_diff * 3.6
        vuln_heavy = (
            (cls1 in self.cfg.vulnerable_classes and cls2 in self.cfg.heavy_classes)
            or (cls2 in self.cfg.vulnerable_classes and cls1 in self.cfg.heavy_classes)
        )
        if vuln_heavy or diff_kmh > 60:
            s = 5
        elif diff_kmh > 40:
            s = 4
        elif diff_kmh > 25:
            s = 3
        elif diff_kmh > 10:
            s = 2
        else:
            s = 1

        score = p * s
        label = "High" if score >= 15 else ("Medium" if score >= 8 else "Low")
        return label, p, s, int(score)

    def evaluate(self, frame_idx: int, timestamp: str, tracks: Dict[int, TrackState]) -> List[dict]:
        incidents = []
        ids = list(tracks.keys())

        for k in list(self.pair_cooldown):
            self.pair_cooldown[k] -= 1
            if self.pair_cooldown[k] <= 0:
                del self.pair_cooldown[k]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                t1, t2 = tracks[id1], tracks[id2]
                key = self._pair_key(id1, id2)

                if t1.age < self.cfg.min_track_age_for_risk or t2.age < self.cfg.min_track_age_for_risk:
                    self.pair_persist[key] = 0
                    continue
                if key in self.pair_cooldown:
                    continue

                p1, p2 = t1.positions[-1], t2.positions[-1]
                v1, v2 = t1.vel, t2.vel
                d = float(np.linalg.norm(p2 - p1))
                if d > self.cfg.proximity_gate_m:
                    self.pair_persist[key] = 0
                    continue

                rel = p2 - p1
                rel_dir = rel / (np.linalg.norm(rel) + 1e-6)
                closing = float(np.dot(v1 - v2, rel_dir))
                if closing < self.cfg.min_closing_speed_mps:
                    self.pair_persist[key] = 0
                    continue

                hdeg = self._heading_conflict_deg(v1, v2)
                if hdeg < self.cfg.heading_conflict_deg and closing < 1.0:
                    self.pair_persist[key] = 0
                    continue

                ttc, dnow = self._ttc_cv(p1, v1, p2, v2, self.cfg.ttc_max_eval_s)
                if not np.isfinite(ttc):
                    self.pair_persist[key] = 0
                    continue

                self.pair_persist[key] += 1
                if self.pair_persist[key] < self.cfg.incident_persist_frames:
                    continue

                s1, s2 = t1.speed, t2.speed
                risk, prob, sev, score = self._risk(ttc, dnow, abs(s1 - s2), t1.cls_name, t2.cls_name)
                if dnow <= self.cfg.collision_radius_m and ttc <= 1.0 and score < 15:
                    risk, score = "High", 15

                incidents.append({
                    "timestamp": timestamp,
                    "frame": frame_idx,
                    "type": "Near-Miss Collision",
                    "actor_1": f"{t1.cls_name} (ID:{id1})",
                    "actor_2": f"{t2.cls_name} (ID:{id2})",
                    "distance_m": round(dnow, 2),
                    "ttc_s": round(ttc, 2),
                    "speed_diff_mps": round(abs(s1 - s2), 2),
                    "speed_diff_kmh": round(abs(s1 - s2) * 3.6, 1),
                    "probability_level": prob,
                    "severity_level": sev,
                    "composite_score": score,
                    "risk": risk
                })

                self.pair_cooldown[key] = self.cfg.incident_cooldown_frames
                self.pair_persist[key] = 0

        # hard braking
        for tid, trk in tracks.items():
            if trk.age < self.cfg.min_track_age_for_risk:
                continue
            if trk.speed >= self.cfg.hard_brake_min_speed_mps and trk.acc < self.cfg.hard_brake_mps2:
                incidents.append({
                    "timestamp": timestamp,
                    "frame": frame_idx,
                    "type": "Hard Braking",
                    "actor_1": f"{trk.cls_name} (ID:{tid})",
                    "actor_2": None,
                    "distance_m": None,
                    "ttc_s": None,
                    "speed_mps": round(trk.speed, 2),
                    "acceleration_mps2": round(trk.acc, 2),
                    "risk": "Medium"
                })

        return incidents