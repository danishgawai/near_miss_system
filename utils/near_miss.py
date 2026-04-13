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
        self.max_heading_delta_deg = float(getattr(cfg, "max_heading_delta_deg", 35.0))
        self.max_lateral_sep_m = float(getattr(cfg, "max_lateral_sep_m", 4.0))
        self.min_parallel_speed_mps = float(getattr(cfg, "min_parallel_speed_mps", 2.0))

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _safe_norm(v: np.ndarray):
        n = float(np.linalg.norm(v))
        return v / (n + 1e-6), n

    @classmethod
    def _heading_deg(cls, v1: np.ndarray, v2: np.ndarray) -> float:
        a, n1 = cls._safe_norm(v1)
        b, n2 = cls._safe_norm(v2)
        if n1 < cls._eps_speed() or n2 < cls._eps_speed():
            return 999.0
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    @staticmethod
    def _eps_speed() -> float:
        return 1.5

    @classmethod
    def _road_axis(cls, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        axis = v1 + v2
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            axis = v1 if np.linalg.norm(v1) >= np.linalg.norm(v2) else v2
        u, _ = cls._safe_norm(axis.astype(np.float32))
        return u.astype(np.float32)

    @classmethod
    def _lane_geometry(cls, pos1, vel1, pos2, vel2):
        u = cls._road_axis(vel1, vel2)
        n = np.array([-u[1], u[0]], dtype=np.float32)
        dp = pos2 - pos1
        dv = vel1 - vel2

        signed_long = float(np.dot(dp, u))
        long_sep = abs(signed_long)
        lat_sep = abs(float(np.dot(dp, n)))

        closing = float(np.dot(dv, u))
        if signed_long < 0:
            closing = -closing

        return long_sep, lat_sep, closing, u

    @staticmethod
    def _ttc_along_axis(long_sep: float, closing_speed: float, tmax: float) -> float:
        if long_sep < 1e-6 or closing_speed <= 1e-6:
            return float("inf")
        ttc = long_sep / closing_speed
        return ttc if 0.0 <= ttc <= tmax else float("inf")

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

        if dist <= self.cfg.collision_radius_m and ttc <= self.cfg.ttc_high_s:
            p = max(p, 5)
            s = max(s, 3)

        score = int(p * s)
        label = "High" if score >= 15 else ("Medium" if score >= 8 else "Low")
        return label, p, s, score

    def evaluate(self, frame_idx: int, timestamp: str, tracks: Dict[int, TrackState]) -> List[dict]:
        incidents = []
        ids = list(tracks.keys())

        for key in list(self.pair_cooldown.keys()):
            self.pair_cooldown[key] -= 1
            if self.pair_cooldown[key] <= 0:
                del self.pair_cooldown[key]

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
                if len(t1.positions) < 2 or len(t2.positions) < 2:
                    self.pair_persist[key] = 0
                    continue

                p1, p2 = t1.positions[-1], t2.positions[-1]
                v1, v2 = t1.vel, t2.vel
                d = float(np.linalg.norm(p2 - p1))
                if d > self.cfg.proximity_gate_m:
                    self.pair_persist[key] = 0
                    continue

                heading_delta = self._heading_deg(v1, v2)
                if heading_delta > self.max_heading_delta_deg:
                    self.pair_persist[key] = 0
                    continue

                s1, s2 = float(t1.speed), float(t2.speed)
                if min(s1, s2) < self.min_parallel_speed_mps:
                    self.pair_persist[key] = 0
                    continue

                long_sep, lat_sep, closing, _ = self._lane_geometry(p1, v1, p2, v2)
                if lat_sep > self.max_lateral_sep_m:
                    self.pair_persist[key] = 0
                    continue
                if closing < self.cfg.min_closing_speed_mps:
                    self.pair_persist[key] = 0
                    continue

                ttc = self._ttc_along_axis(long_sep, closing, self.cfg.ttc_max_eval_s)
                if not np.isfinite(ttc):
                    self.pair_persist[key] = 0
                    continue

                self.pair_persist[key] += 1
                if self.pair_persist[key] < self.cfg.incident_persist_frames:
                    continue

                risk, prob, sev, score = self._risk(ttc, d, abs(s1 - s2), t1.cls_name, t2.cls_name)
                incidents.append({
                    "timestamp": timestamp,
                    "frame": frame_idx,
                    "type": "Near-Miss Collision",
                    "actor_1": f"{t1.cls_name} (ID:{id1})",
                    "actor_2": f"{t2.cls_name} (ID:{id2})",
                    "distance_m": round(d, 2),
                    "ttc_s": round(ttc, 2),
                    "longitudinal_sep_m": round(long_sep, 2),
                    "lateral_sep_m": round(lat_sep, 2),
                    "heading_delta_deg": round(heading_delta, 1),
                    "closing_speed_mps": round(closing, 2),
                    "speed_diff_mps": round(abs(s1 - s2), 2),
                    "speed_diff_kmh": round(abs(s1 - s2) * 3.6, 1),
                    "probability_level": prob,
                    "severity_level": sev,
                    "composite_score": score,
                    "risk": risk,
                })

                self.pair_cooldown[key] = self.cfg.incident_cooldown_frames
                self.pair_persist[key] = 0

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
                    "risk": "Medium",
                })

        return incidents