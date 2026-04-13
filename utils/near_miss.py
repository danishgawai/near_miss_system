import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from utils.motion import TrackState
except ImportError:
    from motion import TrackState


class NearMissEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pair_persist = defaultdict(int)
        self.pair_cooldown = defaultdict(int)

        self.max_heading_delta_deg = float(
            getattr(cfg, "max_heading_delta_deg", getattr(cfg, "heading_conflict_deg", 35.0))
        )
        self.max_lateral_sep_m = float(getattr(cfg, "max_lateral_sep_m", 4.0))
        self.min_parallel_speed_mps = float(getattr(cfg, "min_parallel_speed_mps", 2.0))
        self.max_bumper_gap_m = float(
            getattr(cfg, "max_bumper_gap_m", getattr(cfg, "proximity_gate_m", 12.0))
        )
        self.min_longitudinal_gap_m = float(getattr(cfg, "min_longitudinal_gap_m", -0.6))
        self.min_composite_score = int(getattr(cfg, "min_composite_score", 3))

    # -----------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _safe_unit(v: np.ndarray) -> Tuple[np.ndarray, float]:
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32), 0.0
        return (v / n).astype(np.float32), n

    # -----------------------------------------------------------------
    # Heading comparison
    # -----------------------------------------------------------------
    @classmethod
    def _heading_deg(cls, v1: np.ndarray, v2: np.ndarray) -> float:
        a, n1 = cls._safe_unit(v1)
        b, n2 = cls._safe_unit(v2)
        # Unknown heading = incompatible; let speed gate handle separately
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    # -----------------------------------------------------------------
    # Road axis (sign-stable, anchored to faster vehicle)
    # -----------------------------------------------------------------
    @classmethod
    def _road_axis(cls, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        # Anchor to faster vehicle for sign stability across frames
        if float(np.linalg.norm(v1)) >= float(np.linalg.norm(v2)):
            anchor = v1
        else:
            anchor = v2

        axis = v1 + v2
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            axis = anchor.copy()

        u, _ = cls._safe_unit(axis.astype(np.float32))

        # Ensure consistent sign with anchor direction
        if float(np.dot(u, anchor)) < 0:
            u = -u

        return u.astype(np.float32)

    # -----------------------------------------------------------------
    # Corridor compatibility
    # -----------------------------------------------------------------
    @staticmethod
    def _corridor_compatible(t1: TrackState, t2: TrackState) -> bool:
        c1 = getattr(t1, "corridor_id", None)
        c2 = getattr(t2, "corridor_id", None)
        if c1 is None or c2 is None:
            return True
        return c1 == c2

    # -----------------------------------------------------------------
    # Front/back ordering along axis
    # -----------------------------------------------------------------
    @staticmethod
    def _split_front_back(pos1: np.ndarray, pos2: np.ndarray, u: np.ndarray):
        s1 = float(np.dot(pos1, u))
        s2 = float(np.dot(pos2, u))
        return (1, 2) if s1 >= s2 else (2, 1)

    # -----------------------------------------------------------------
    # Bumper-to-bumper gap
    # -----------------------------------------------------------------
    @staticmethod
    def _bumper_gap_metrics(
        front_obj: TrackState,
        back_obj: TrackState,
        u: np.ndarray,
        front_center: np.ndarray,
        back_center: np.ndarray,
    ):
        if front_obj.has_extent and back_obj.has_extent:
            p_front_rear = front_obj.rear_bev
            p_back_front = back_obj.front_bev
        else:
            p_front_rear = front_center
            p_back_front = back_center

        delta = p_front_rear - p_back_front
        gap_euclid = float(np.linalg.norm(delta))
        gap_long = float(np.dot(delta, u))
        return gap_euclid, gap_long, p_front_rear, p_back_front

    # -----------------------------------------------------------------
    # TTC from longitudinal gap
    # -----------------------------------------------------------------
    @staticmethod
    def _ttc_from_gap(gap_long: float, closing_speed: float, tmax: float) -> float:
        if gap_long <= 1e-6 or closing_speed <= 1e-6:
            return float("inf")
        ttc = gap_long / closing_speed
        return ttc if 0.0 <= ttc <= tmax else float("inf")

    # -----------------------------------------------------------------
    # 5x5 risk matrix
    # -----------------------------------------------------------------
    def _risk(self, ttc, dist, speed_diff, cls1, cls2):
        # Probability from TTC + distance
        if ttc <= self.cfg.ttc_high_s:
            probability = 5
        elif ttc <= self.cfg.ttc_med_s:
            probability = 4
        elif ttc <= self.cfg.ttc_low_s:
            probability = 3
        elif dist <= self.cfg.proximity_gate_m * 0.5:
            probability = 2
        else:
            probability = 1

        # Severity from class interaction + speed delta
        diff_kmh = speed_diff * 3.6
        vuln_heavy = (
            (cls1 in self.cfg.vulnerable_classes and cls2 in self.cfg.heavy_classes)
            or (cls2 in self.cfg.vulnerable_classes and cls1 in self.cfg.heavy_classes)
        )

        if vuln_heavy or diff_kmh > 60:
            severity = 5
        elif diff_kmh > 40:
            severity = 4
        elif diff_kmh > 25:
            severity = 3
        elif diff_kmh > 10:
            severity = 2
        else:
            severity = 1

        # Imminent close-contact boost
        if dist <= self.cfg.collision_radius_m and ttc <= self.cfg.ttc_high_s:
            probability = max(probability, 5)
            severity = max(severity, 3)

        score = int(probability * severity)
        label = "High" if score >= 15 else ("Medium" if score >= 8 else "Low")
        return label, probability, severity, score

    # -----------------------------------------------------------------
    # Main per-frame evaluation
    # -----------------------------------------------------------------
    def evaluate(
        self, frame_idx: int, timestamp: str, tracks: Dict[int, TrackState]
    ) -> List[dict]:
        incidents = []
        ids = list(tracks.keys())

        # Cooldown decay
        for key in list(self.pair_cooldown.keys()):
            self.pair_cooldown[key] -= 1
            if self.pair_cooldown[key] <= 0:
                del self.pair_cooldown[key]

        # ---- Pairwise evaluation ----
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                t1, t2 = tracks[id1], tracks[id2]
                key = self._pair_key(id1, id2)

                # Gate 1: track maturity
                if (
                    t1.age < self.cfg.min_track_age_for_risk
                    or t2.age < self.cfg.min_track_age_for_risk
                ):
                    self.pair_persist[key] = 0
                    continue

                # Gate 2: cooldown
                if key in self.pair_cooldown:
                    continue

                # Gate 3: position history
                if len(t1.positions) < 2 or len(t2.positions) < 2:
                    self.pair_persist[key] = 0
                    continue

                # Gate 4: carriageway / corridor
                if not self._corridor_compatible(t1, t2):
                    self.pair_persist[key] = 0
                    continue

                p1, p2 = t1.positions[-1], t2.positions[-1]
                v1, v2 = t1.vel, t2.vel

                # Gate 5: coarse center distance
                center_dist = float(np.linalg.norm(p2 - p1))
                if center_dist > self.cfg.proximity_gate_m:
                    self.pair_persist[key] = 0
                    continue

                # Gate 6: heading compatibility
                heading_delta = self._heading_deg(v1, v2)
                if heading_delta > self.max_heading_delta_deg:
                    self.pair_persist[key] = 0
                    continue

                # Gate 7: minimum speed for both tracks
                s1, s2 = float(t1.speed), float(t2.speed)
                if min(s1, s2) < self.min_parallel_speed_mps:
                    self.pair_persist[key] = 0
                    continue

                # Pair road axis
                u = self._road_axis(v1, v2)
                n = np.array([-u[1], u[0]], dtype=np.float32)

                # Gate 8: lateral separation
                dp = p2 - p1
                lat_sep = abs(float(np.dot(dp, n)))
                if lat_sep > self.max_lateral_sep_m:
                    self.pair_persist[key] = 0
                    continue

                # Determine front/back ordering
                front_idx, _ = self._split_front_back(p1, p2, u)
                if front_idx == 1:
                    front_obj, back_obj = t1, t2
                    front_id, back_id = id1, id2
                    front_center, back_center = p1, p2
                else:
                    front_obj, back_obj = t2, t1
                    front_id, back_id = id2, id1
                    front_center, back_center = p2, p1

                # Gate 9: bumper-to-bumper gap
                bumper_gap, gap_long, rear_pt, front_pt = self._bumper_gap_metrics(
                    front_obj, back_obj, u, front_center, back_center
                )
                if bumper_gap > self.max_bumper_gap_m:
                    self.pair_persist[key] = 0
                    continue
                if gap_long < self.min_longitudinal_gap_m:
                    self.pair_persist[key] = 0
                    continue

                # Gate 10: vector closing speed along road axis
                v_front_long = float(np.dot(front_obj.vel, u))
                v_back_long = float(np.dot(back_obj.vel, u))
                closing = max(0.0, v_back_long - v_front_long)
                if closing < self.cfg.min_closing_speed_mps:
                    self.pair_persist[key] = 0
                    continue

                # TTC from longitudinal bumper gap
                ttc = self._ttc_from_gap(
                    max(gap_long, 0.0), closing, self.cfg.ttc_max_eval_s
                )
                if not np.isfinite(ttc):
                    self.pair_persist[key] = 0
                    continue

                # Gate 11: temporal persistence
                self.pair_persist[key] += 1
                if self.pair_persist[key] < self.cfg.incident_persist_frames:
                    continue

                # Risk scoring
                risk, probability, severity, score = self._risk(
                    ttc=ttc,
                    dist=bumper_gap,
                    speed_diff=abs(s1 - s2),
                    cls1=t1.cls_name,
                    cls2=t2.cls_name,
                )

                # Gate 12: minimum composite score
                if score < self.min_composite_score:
                    self.pair_persist[key] = 0
                    continue

                # Build incident record
                inc = {
                    "timestamp": timestamp,
                    "frame": frame_idx,
                    "type": "Near-Miss Collision",
                    "actor_1": f"{back_obj.cls_name} (ID:{back_id})",
                    "actor_2": f"{front_obj.cls_name} (ID:{front_id})",
                    # Primary metrics (bumper-based)
                    "distance_m": round(bumper_gap, 2),
                    "ttc_s": round(ttc, 2),
                    # Diagnostics
                    "longitudinal_gap_m": round(float(gap_long), 2),
                    "lateral_sep_m": round(lat_sep, 2),
                    "heading_delta_deg": round(heading_delta, 1),
                    "closing_speed_mps": round(closing, 2),
                    "front_long_speed_mps": round(v_front_long, 2),
                    "back_long_speed_mps": round(v_back_long, 2),
                    # Rear-front pair detail
                    "rear_front_pair": {
                        "front_vehicle_id": int(front_id),
                        "back_vehicle_id": int(back_id),
                        "rear_point_front_vehicle_m": [
                            round(float(rear_pt[0]), 3),
                            round(float(rear_pt[1]), 3),
                        ],
                        "front_point_back_vehicle_m": [
                            round(float(front_pt[0]), 3),
                            round(float(front_pt[1]), 3),
                        ],
                    },
                    # Speed
                    "speed_diff_mps": round(abs(s1 - s2), 2),
                    "speed_diff_kmh": round(abs(s1 - s2) * 3.6, 1),
                    # Risk breakdown
                    "probability_level": int(probability),
                    "severity_level": int(severity),
                    "composite_score": int(score),
                    "risk": risk,
                }

                # Only include corridor_id when actually assigned
                cid = getattr(back_obj, "corridor_id", None)
                if cid is not None:
                    inc["corridor_id"] = cid

                incidents.append(inc)
                self.pair_cooldown[key] = self.cfg.incident_cooldown_frames
                self.pair_persist[key] = 0

        # ---- Hard-braking events ----
        for tid, trk in tracks.items():
            if trk.age < self.cfg.min_track_age_for_risk:
                continue
            if (
                trk.speed >= self.cfg.hard_brake_min_speed_mps
                and trk.acc < self.cfg.hard_brake_mps2
            ):
                incidents.append(
                    {
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
                    }
                )

        return incidents