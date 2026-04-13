import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List
from utils.motion import TrackState


class NearMissEngine:
    """
    Simple but robust POC logic:
    - Pair risk via 2D CPA TTC (+ optional 1D rear-end TTC assist)
    - Scenario classification by heading delta
    - Emergency distance override
    - Persistence + cooldown
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.pair_persist = defaultdict(int)
        self.pair_cooldown = defaultdict(int)

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _safe_norm(v: np.ndarray):
        n = float(np.linalg.norm(v))
        return v / (n + 1e-6), n

    def _heading_delta_deg(self, v1: np.ndarray, v2: np.ndarray) -> float:
        a, n1 = self._safe_norm(v1)
        b, n2 = self._safe_norm(v2)
        if n1 < 0.5 or n2 < 0.5:
            return 999.0
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _classify_scenario(self, heading_delta_deg: float) -> str:
        if heading_delta_deg <= self.cfg.heading_rear_end_max_deg:
            return "rear_end"
        if heading_delta_deg <= self.cfg.heading_merging_max_deg:
            return "merging"
        if heading_delta_deg <= self.cfg.heading_crossing_max_deg:
            return "crossing"
        return "head_on"

    @staticmethod
    def _ttc_cpa(p1, v1, p2, v2, tmax):
        dp = p2 - p1
        dv = v2 - v1
        dv_sq = float(np.dot(dv, dv))
        if dv_sq < 1e-8:
            return float("inf"), float(np.linalg.norm(dp))
        t = -float(np.dot(dp, dv)) / dv_sq
        t = float(np.clip(t, 0.0, tmax))
        closest = dp + dv * t
        dmin = float(np.linalg.norm(closest))
        return t, dmin

    @staticmethod
    def _ttc_1d(long_sep: float, closing_speed: float, tmax: float):
        if long_sep <= 1e-6 or closing_speed <= 1e-6:
            return float("inf")
        t = long_sep / closing_speed
        return t if 0.0 <= t <= tmax else float("inf")

    def _axis_geometry(self, p1, v1, p2, v2):
        axis = v1 + v2
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = v1 if float(np.linalg.norm(v1)) >= float(np.linalg.norm(v2)) else v2
        u, _ = self._safe_norm(axis.astype(np.float32))
        n = np.array([-u[1], u[0]], dtype=np.float32)

        dp = p2 - p1
        dv = v1 - v2
        signed_long = float(np.dot(dp, u))
        long_sep = abs(signed_long)
        lat_sep = abs(float(np.dot(dp, n)))
        closing = float(np.dot(dv, u))
        if signed_long < 0:
            closing = -closing
        return long_sep, lat_sep, closing

    def _risk(self, ttc, dist, speed_diff, closing_speed, cls1, cls2, scenario):
        # Emergency override: near collision zone
        if dist <= self.cfg.collision_radius_m * 1.15:
            return "High", 5, 5, 25

        # scenario scales
        scen_scale = {
            "rear_end": 1.0, "merging": 1.0, "crossing": 1.1, "head_on": 1.1
        }.get(scenario, 1.0)

        ttc_ref = max(0.2, self.cfg.ttc_low_s * scen_scale)
        ttc_score = 1.0 - min(ttc / ttc_ref, 1.0) if np.isfinite(ttc) else 0.0

        dist_ref = max(self.cfg.collision_radius_m * 2.5, 0.5)
        dist_score = 1.0 - min(dist / dist_ref, 1.0)

        rel_score = min(speed_diff / 15.0, 1.0)
        closing_score = min(max(closing_speed, 0.0) / 12.0, 1.0)

        vuln_heavy = (
            (cls1 in self.cfg.vulnerable_classes and cls2 in self.cfg.heavy_classes) or
            (cls2 in self.cfg.vulnerable_classes and cls1 in self.cfg.heavy_classes)
        )
        vh_boost = 1.08 if vuln_heavy else 1.0
        scen_boost = 1.15 if scenario == "head_on" else (1.08 if scenario == "crossing" else 1.0)

        danger = (
            0.40 * ttc_score +
            0.35 * dist_score +
            0.15 * rel_score +
            0.10 * closing_score
        ) * vh_boost * scen_boost
        danger = float(np.clip(danger, 0.0, 1.0))

        if danger >= 0.78:
            label = "High"
        elif danger >= 0.48:
            label = "Medium"
        else:
            label = "Low"

        p = int(np.clip(round(1 + 4 * (0.55 * ttc_score + 0.45 * dist_score)), 1, 5))
        s = int(np.clip(round(1 + 4 * (0.6 * rel_score + 0.4 * closing_score)), 1, 5))
        score = p * s
        return label, p, s, score

    def evaluate(self, frame_idx: int, timestamp: str, tracks: Dict[int, TrackState]) -> List[dict]:
        incidents = []
        ids = list(tracks.keys())

        # cooldown tick
        for key in list(self.pair_cooldown.keys()):
            self.pair_cooldown[key] -= 1
            if self.pair_cooldown[key] <= 0:
                del self.pair_cooldown[key]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                t1, t2 = tracks[id1], tracks[id2]
                key = self._pair_key(id1, id2)

                if key in self.pair_cooldown:
                    continue
                if t1.age < self.cfg.min_track_age_for_risk or t2.age < self.cfg.min_track_age_for_risk:
                    self.pair_persist[key] = 0
                    continue
                if len(t1.bev_positions) < 2 or len(t2.bev_positions) < 2:
                    self.pair_persist[key] = 0
                    continue

                p1, p2 = t1.bev_positions[-1], t2.bev_positions[-1]
                v1, v2 = t1.vel, t2.vel
                d = float(np.linalg.norm(p2 - p1))
                if d > self.cfg.proximity_gate_m:
                    self.pair_persist[key] = 0
                    continue

                # soft quality penalty (do not hard reject if very close)
                close_zone = d <= self.cfg.collision_radius_m * 1.8
                quality_penalty = 1.0
                min_conf = min(t1.confidence, t2.confidence)
                min_dir = min(t1.direction_consistency, t2.direction_consistency)

                if min_conf < self.cfg.min_confidence_for_risk and not close_zone:
                    self.pair_persist[key] = 0
                    continue
                elif min_conf < self.cfg.min_confidence_for_risk:
                    quality_penalty *= 0.90

                if min_dir < self.cfg.direction_consistency_min and not close_zone:
                    self.pair_persist[key] = 0
                    continue
                elif min_dir < self.cfg.direction_consistency_min:
                    quality_penalty *= 0.92

                heading_delta = self._heading_delta_deg(v1, v2)
                scenario = self._classify_scenario(heading_delta)

                long_sep, lat_sep, closing = self._axis_geometry(p1, v1, p2, v2)

                # Lateral gating for aligned scenarios only
                lat_limit = 4.0 if scenario == "rear_end" else (5.0 if scenario == "merging" else 8.0)
                if scenario in ("rear_end", "merging") and lat_sep > lat_limit:
                    self.pair_persist[key] = 0
                    continue

                # TTC via CPA
                t_cpa, d_cpa = self._ttc_cpa(p1, v1, p2, v2, self.cfg.ttc_max_eval_s)
                if d_cpa > self.cfg.cpa_miss_distance_m and not close_zone:
                    self.pair_persist[key] = 0
                    continue

                ttc = t_cpa
                if scenario == "rear_end" and closing > self.cfg.min_closing_speed_mps:
                    ttc_1d = self._ttc_1d(long_sep, closing, self.cfg.ttc_max_eval_s)
                    ttc = min(ttc, ttc_1d)

                if not np.isfinite(ttc):
                    self.pair_persist[key] = 0
                    continue

                self.pair_persist[key] += 1
                if self.pair_persist[key] < self.cfg.incident_persist_frames:
                    continue

                risk, prob, sev, score = self._risk(
                    ttc=ttc,
                    dist=d,
                    speed_diff=abs(float(t1.speed - t2.speed)),
                    closing_speed=abs(closing),
                    cls1=t1.cls_name,
                    cls2=t2.cls_name,
                    scenario=scenario
                )

                score = int(round(score * quality_penalty))
                if score >= 15:
                    risk = "High"
                elif score >= 8:
                    risk = "Medium"
                else:
                    risk = "Low"

                incidents.append({
                    "timestamp": timestamp,
                    "frame": frame_idx,
                    "type": f"Near-Miss ({scenario.replace('_', ' ').title()})",
                    "scenario": scenario,
                    "actor_1": f"{t1.cls_name} (ID:{id1})",
                    "actor_2": f"{t2.cls_name} (ID:{id2})",
                    "distance_m": round(d, 2),
                    "ttc_s": round(float(ttc), 2),
                    "longitudinal_sep_m": round(long_sep, 2),
                    "lateral_sep_m": round(lat_sep, 2),
                    "heading_delta_deg": round(heading_delta, 1),
                    "closing_speed_mps": round(closing, 2),
                    "speed_diff_mps": round(abs(float(t1.speed - t2.speed)), 2),
                    "speed_diff_kmh": round(abs(float(t1.speed - t2.speed)) * 3.6, 1),
                    "probability_level": int(prob),
                    "severity_level": int(sev),
                    "composite_score": int(score),
                    "risk": risk,
                    "confidence": round(float(min_conf), 3),
                })

                self.pair_cooldown[key] = self.cfg.incident_cooldown_frames
                self.pair_persist[key] = 0

        # Single-object events
        for tid, trk in tracks.items():
            if trk.age < self.cfg.min_track_age_for_risk:
                continue

            if trk.speed >= self.cfg.hard_brake_min_speed_mps and trk.acc < self.cfg.hard_brake_mps2:
                incidents.append({
                    "timestamp": timestamp,
                    "frame": frame_idx,
                    "type": "Hard Braking",
                    "scenario": "hard_brake",
                    "actor_1": f"{trk.cls_name} (ID:{tid})",
                    "actor_2": None,
                    "distance_m": None,
                    "ttc_s": None,
                    "speed_mps": round(float(trk.speed), 2),
                    "acceleration_mps2": round(float(trk.acc), 2),
                    "risk": "Medium",
                    "confidence": round(float(trk.confidence), 3),
                })

            if trk.speed >= self.cfg.swerve_min_speed_mps and trk.lateral_acc >= self.cfg.swerve_lateral_acc_threshold:
                incidents.append({
                    "timestamp": timestamp,
                    "frame": frame_idx,
                    "type": "Sudden Swerve",
                    "scenario": "swerve",
                    "actor_1": f"{trk.cls_name} (ID:{tid})",
                    "actor_2": None,
                    "distance_m": None,
                    "ttc_s": None,
                    "speed_mps": round(float(trk.speed), 2),
                    "lateral_acc_mps2": round(float(trk.lateral_acc), 2),
                    "risk": "Medium",
                    "confidence": round(float(trk.confidence), 3),
                })

        return incidents