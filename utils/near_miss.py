"""
Near-miss detection engine.

Gate order (applied strictly before risk scoring):
  1.  Cooldown          — pair recently triggered, skip
  2.  Track age         — velocity EMA not yet stable
  3.  Frame border      — BEV unreliable at frame edges
  4.  BEV positions     — need ≥2 for velocity
  5.  Proximity         — coarse centre-to-centre filter
  6.  Minimum speed     — cannot compute heading if both stopped
  7.  Heading validity  — sentinel 999 → skip
  8.  Scenario + lateral gate — applied to ALL scenarios
  9.  Closing speed     — reject diverging pairs
  10. Quality           — confidence + direction consistency
  11. CPA miss distance — trajectories won't conflict
  12. Persistence       — must flag N consecutive frames
  13. RI scoring        — RI = α·R_d + β·R_ttc + γ·R_v
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List

from utils.motion import TrackState


class NearMissEngine:

    def __init__(self, cfg):
        self.cfg = cfg
        self.pair_persist: Dict[Tuple[int, int], int] = defaultdict(int)
        self.pair_cooldown: Dict[Tuple[int, int], int] = defaultdict(int)
        self.swerve_persist: Dict[int, int] = defaultdict(int)

    # ── static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _safe_norm(v: np.ndarray) -> Tuple[np.ndarray, float]:
        n = float(np.linalg.norm(v))
        return v / (n + 1e-6), n

    # ── spatial helpers ───────────────────────────────────────────────────

    def _approx_bev_half_diag(self, trk: TrackState) -> float:
        """
        Approximate half-diagonal of a vehicle's footprint in BEV metres.
        Derived from the image bbox scaled by pixels-per-metre.
        Used to convert centre-to-centre distance to edge-to-edge.
        """
        if not trk.last_bbox:
            return 1.0  # conservative 1 m fallback
        x1, y1, x2, y2 = trk.last_bbox
        w_px = max(1.0, float(x2 - x1))
        h_px = max(1.0, float(y2 - y1))
        ppm  = max(self.cfg.default_pixels_per_meter, 1.0)
        w_m  = w_px / ppm
        h_m  = h_px / ppm
        # Half-diagonal of the bbox rectangle in BEV space
        return 0.5 * float(np.hypot(w_m, h_m))

    def _edge_to_edge(self, t1: TrackState, t2: TrackState) -> float:
        """
        Edge-to-edge BEV distance between two tracked objects.
        = max(0,  centre_dist  −  r1  −  r2)
        where r1, r2 are the approximate half-diagonals of their bboxes.
        """
        p1 = np.array(t1.bev_positions[-1], dtype=np.float32)
        p2 = np.array(t2.bev_positions[-1], dtype=np.float32)
        centre_dist = float(np.linalg.norm(p2 - p1))
        r1 = self._approx_bev_half_diag(t1)
        r2 = self._approx_bev_half_diag(t2)
        return max(0.0, centre_dist - r1 - r2)

    def _is_near_border(self, trk: TrackState) -> bool:
        if not trk.last_bbox:
            return False
        x1, y1, x2, y2 = trk.last_bbox
        m  = self.cfg.frame_border_margin
        fw = self.cfg.frame_width
        fh = self.cfg.frame_height
        return x1 < m or y1 < m or x2 > fw - m or y2 > fh - m

    # ── heading / scenario ────────────────────────────────────────────────

    def _heading_delta_deg(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Angle between two velocity vectors (degrees).
        Returns sentinel 999.0 if either track is too slow for a valid heading.
        """
        _, n1 = self._safe_norm(v1)
        _, n2 = self._safe_norm(v2)
        if n1 < self.cfg.min_speed_for_risk_mps or n2 < self.cfg.min_speed_for_risk_mps:
            return 999.0
        a, _ = self._safe_norm(v1)
        b, _ = self._safe_norm(v2)
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

    def _lat_limit(self, scenario: str) -> float:
        return {
            "rear_end": self.cfg.lat_limit_rear_end,
            "merging":  self.cfg.lat_limit_merging,
            "crossing": self.cfg.lat_limit_crossing,
            "head_on":  self.cfg.lat_limit_head_on,
        }.get(scenario, self.cfg.lat_limit_crossing)

    # ── geometry ──────────────────────────────────────────────────────────

    def _geometry(
        self,
        p1: np.ndarray,
        v1: np.ndarray,
        p2: np.ndarray,
        v2: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Returns (long_sep, lat_sep, closing) all in metres / m·s⁻¹.

        long_sep : Euclidean centre-to-centre distance (m)
        lat_sep  : predicted lateral miss distance at closest approach (m)
                   = min lateral velocity component × estimated time-to-close
                   Tells us whether the paths will actually intersect or pass
                   side-by-side.
        closing  : rate of gap reduction along the separation axis (m/s)
                   positive = approaching, negative = diverging
        """
        dp   = p2 - p1
        dist = float(np.linalg.norm(dp))
        if dist < 1e-6:
            return 0.0, 0.0, 0.0

        u_sep = (dp / dist).astype(np.float32)           # unit p1→p2
        u_lat = np.array([-u_sep[1], u_sep[0]], dtype=np.float32)

        long_sep = dist

        # Closing speed along the separation axis
        dv      = v1 - v2
        closing = float(np.dot(dv, u_sep))

        # Predicted lateral miss distance (metres):
        # Each vehicle's lateral velocity component × time-to-close
        t_close = dist / max(abs(closing), 0.1)          # seconds
        t_close = min(t_close, self.cfg.ttc_max_eval_s)  # cap at eval window

        v1_lat  = abs(float(np.dot(v1, u_lat)))          # m/s
        v2_lat  = abs(float(np.dot(v2, u_lat)))          # m/s
        # Conservative estimate: use the smaller lateral deviation
        lat_sep = min(v1_lat, v2_lat) * t_close          # metres ✅

        return long_sep, lat_sep, closing

    # ── TTC ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ttc_cpa(p1, v1, p2, v2, tmax: float) -> Tuple[float, float]:
        dp    = p2 - p1
        dv    = v2 - v1
        dv_sq = float(np.dot(dv, dv))
        if dv_sq < 1e-8:
            return float("inf"), float(np.linalg.norm(dp))
        t     = float(np.clip(-float(np.dot(dp, dv)) / dv_sq, 0.0, tmax))
        d_min = float(np.linalg.norm(dp + dv * t))
        return t, d_min

    @staticmethod
    def _ttc_1d(long_sep: float, closing: float, tmax: float) -> float:
        if long_sep <= 1e-6 or closing <= 1e-6:
            return float("inf")
        t = long_sep / closing
        return t if 0.0 <= t <= tmax else float("inf")

    def _get_ttc_threshold(self, cls1: str, cls2: str) -> float:
        """Per-class-pair TTC gate threshold (binary pass/fail, not scoring ref)."""
        key = f"{cls1}:{cls2}"
        rev = f"{cls2}:{cls1}"
        return float(
            self.cfg.ttc_threshold_by_pair.get(
                key,
                self.cfg.ttc_threshold_by_pair.get(rev, self.cfg.ttc_low_s),
            )
        )

    # ── risk scoring ─────────────────────────────────────────────────────

    def _score_risk(
        self,
        edge_dist: float,
        ttc: float,
        speed_diff: float,
        cls1: str,
        cls2: str,
    ) -> Tuple[str, int, int, int, float]:
        """
        RI = α·R_d + β·R_ttc + γ·R_v

        R_d   — proximity risk, anchored to ri_distance_ref (not proximity_gate_m)
        R_ttc — temporal risk,  anchored to ri_ttc_ref      (not class-pair threshold)
        R_v   — motion intensity

        Class-pair TTC thresholds are used as a binary gate in evaluate(),
        not as the scoring denominator here.
        """
        r_d   = float(np.clip(
            1.0 - edge_dist / max(self.cfg.ri_distance_ref, 1e-6),
            0.0, 1.0,
        ))
        r_ttc = float(np.clip(
            1.0 - ttc / max(self.cfg.ri_ttc_ref, 1e-6),
            0.0, 1.0,
        )) if np.isfinite(ttc) else 1.0

        r_v   = float(np.clip(
            speed_diff / max(self.cfg.ri_v_max, 1e-6),
            0.0, 1.0,
        ))

        ri = float(np.clip(
            self.cfg.ri_alpha * r_d +
            self.cfg.ri_beta  * r_ttc +
            self.cfg.ri_gamma * r_v,
            0.0, 1.0,
        ))

        # Classify using configurable thresholds
        if ri >= self.cfg.ri_high_threshold:
            risk = "High"
        elif ri >= self.cfg.ri_medium_threshold:
            risk = "Medium"
        else:
            risk = "Low"

        prob  = int(np.clip(round(1 + 4 * max(r_d, r_ttc)), 1, 5))
        sev   = int(np.clip(round(1 + 4 * r_v),              1, 5))
        score = int(np.clip(round(ri * 25),                   1, 25))
        return risk, prob, sev, score, ri

    # ── swerve helpers ────────────────────────────────────────────────────

    def _heading_change_deg(self, headings) -> float:
        if len(headings) < 2:
            return 0.0
        delta = headings[-1] - headings[0]
        while delta >  180: delta -= 360
        while delta < -180: delta += 360
        return abs(delta)

    def _is_swerve(self, trk: TrackState) -> bool:
        n = self.cfg.swerve_eval_frames
        if len(trk.lateral_acc_hist) < n or len(trk.heading_hist) < n:
            return False
        lat_vals = list(trk.lateral_acc_hist)[-n:]
        head_vals = list(trk.heading_hist)[-n:]
        avg_lat        = float(np.mean(lat_vals))
        max_lat        = float(np.max(lat_vals))
        heading_change = self._heading_change_deg(head_vals)
        return (
            trk.speed >= self.cfg.swerve_min_speed_mps
            and avg_lat >= self.cfg.swerve_lateral_acc_threshold
            and max_lat >= self.cfg.swerve_lateral_acc_threshold * 1.1
            and heading_change >= self.cfg.swerve_heading_delta_min_deg
            and trk.direction_consistency >= self.cfg.direction_consistency_min
        )

    # ── main entry point ──────────────────────────────────────────────────

    def evaluate(
        self,
        frame_idx: int,
        timestamp: str,
        tracks: Dict[int, TrackState],
    ) -> List[dict]:
        incidents: List[dict] = []
        ids = list(tracks.keys())

        # Tick cooldown counters
        for key in list(self.pair_cooldown.keys()):
            self.pair_cooldown[key] -= 1
            if self.pair_cooldown[key] <= 0:
                del self.pair_cooldown[key]

        # ── Pairwise near-miss evaluation ─────────────────────────────────
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                t1, t2   = tracks[id1], tracks[id2]
                key      = self._pair_key(id1, id2)

                # Gate 1 — cooldown
                if key in self.pair_cooldown:
                    continue

                # Gate 2 — track age
                if (t1.age < self.cfg.min_track_age_for_risk or
                        t2.age < self.cfg.min_track_age_for_risk):
                    self.pair_persist[key] = 0
                    continue

                # Gate 3 — frame border
                if self._is_near_border(t1) or self._is_near_border(t2):
                    self.pair_persist[key] = 0
                    continue

                # Gate 4 — BEV positions available
                if len(t1.bev_positions) < 2 or len(t2.bev_positions) < 2:
                    self.pair_persist[key] = 0
                    continue

                p1 = np.array(t1.bev_positions[-1], dtype=np.float32)
                p2 = np.array(t2.bev_positions[-1], dtype=np.float32)
                v1 = t1.vel.astype(np.float32)
                v2 = t2.vel.astype(np.float32)

                # Gate 5 — coarse proximity (centre-to-centre, cheap filter)
                centre_dist = float(np.linalg.norm(p2 - p1))
                if centre_dist > self.cfg.proximity_gate_m:
                    self.pair_persist[key] = 0
                    continue

                # Gate 6 — minimum speed
                if (t1.speed < self.cfg.min_speed_for_risk_mps and
                        t2.speed < self.cfg.min_speed_for_risk_mps):
                    self.pair_persist[key] = 0
                    continue

                # Gate 7 — heading validity
                heading_delta = self._heading_delta_deg(v1, v2)
                if heading_delta >= 360.0:
                    self.pair_persist[key] = 0
                    continue

                # Gate 8 — scenario + lateral separation
                scenario = self._classify_scenario(heading_delta)
                long_sep, lat_sep, closing = self._geometry(p1, v1, p2, v2)

                if lat_sep > self._lat_limit(scenario):
                    self.pair_persist[key] = 0
                    continue

                # Gate 9 — closing speed (reject diverging pairs)
                if closing < self.cfg.min_closing_speed_mps:
                    self.pair_persist[key] = 0
                    continue

                # Gate 10 — track quality
                min_conf = min(t1.confidence, t2.confidence)
                min_dir  = min(t1.direction_consistency, t2.direction_consistency)
                if min_conf < self.cfg.min_confidence_for_risk:
                    self.pair_persist[key] = 0
                    continue
                if min_dir < self.cfg.direction_consistency_min:
                    self.pair_persist[key] = 0
                    continue

                # Gate 11 — CPA miss distance
                t_cpa, d_cpa = self._ttc_cpa(
                    p1, v1, p2, v2, self.cfg.ttc_max_eval_s
                )
                if d_cpa > self.cfg.cpa_miss_distance_m:
                    self.pair_persist[key] = 0
                    continue

                # TTC selection
                ttc = t_cpa
                if scenario == "rear_end":
                    ttc_1d = self._ttc_1d(long_sep, closing, self.cfg.ttc_max_eval_s)
                    ttc = min(ttc, ttc_1d)

                if not np.isfinite(ttc):
                    self.pair_persist[key] = 0
                    continue

                # Binary TTC gate — per class pair (separate from RI scoring)
                ttc_gate = self._get_ttc_threshold(t1.cls_name, t2.cls_name)
                if ttc > ttc_gate:
                    self.pair_persist[key] = 0
                    continue

                # Gate 12 — persistence
                self.pair_persist[key] += 1
                if self.pair_persist[key] < self.cfg.incident_persist_frames:
                    continue

                # Gate 13 — RI scoring on EDGE-TO-EDGE distance
                edge_dist  = self._edge_to_edge(t1, t2)
                speed_diff = abs(float(t1.speed - t2.speed))

                risk, prob, sev, score, ri = self._score_risk(
                    edge_dist=edge_dist,
                    ttc=ttc,
                    speed_diff=speed_diff,
                    cls1=t1.cls_name,
                    cls2=t2.cls_name,
                )

                # Vulnerable-heavy boost
                vuln_heavy = (
                    (t1.cls_name in self.cfg.vulnerable_classes and
                     t2.cls_name in self.cfg.heavy_classes) or
                    (t2.cls_name in self.cfg.vulnerable_classes and
                     t1.cls_name in self.cfg.heavy_classes)
                )
                if vuln_heavy:
                    ri = float(np.clip(ri * 1.10, 0.0, 1.0))
                    score = int(np.clip(round(ri * 25), 1, 25))
                    if ri >= self.cfg.ri_high_threshold:
                        risk = "High"
                    elif ri >= self.cfg.ri_medium_threshold:
                        risk = "Medium"

                incidents.append({
                    "timestamp":          timestamp,
                    "frame":              frame_idx,
                    "type":               f"Near-Miss ({scenario.replace('_', ' ').title()})",
                    "scenario":           scenario,
                    "actor_1":            f"{t1.cls_name} (ID:{id1})",
                    "actor_2":            f"{t2.cls_name} (ID:{id2})",
                    "edge_distance_m":    round(edge_dist, 2),
                    "centre_distance_m":  round(centre_dist, 2),
                    "ttc_s":              round(float(ttc), 2),
                    "longitudinal_sep_m": round(long_sep, 2),
                    "lateral_sep_m":      round(lat_sep, 2),
                    "heading_delta_deg":  round(heading_delta, 1),
                    "closing_speed_mps":  round(closing, 2),
                    "speed_diff_mps":     round(speed_diff, 2),
                    "speed_diff_kmh":     round(speed_diff * 3.6, 1),
                    "probability_level":  int(prob),
                    "severity_level":     int(sev),
                    "composite_score":    int(score),
                    "risk_index":         round(ri, 3),
                    "risk":               risk,
                    "confidence":         round(float(min_conf), 3),
                })

                self.pair_cooldown[key] = self.cfg.incident_cooldown_frames
                self.pair_persist[key]  = 0

        # ── Single-object events ──────────────────────────────────────────
        for tid, trk in tracks.items():
            if trk.age < self.cfg.min_track_age_for_risk:
                continue
            if self._is_near_border(trk):
                continue

            # Hard braking
            if (trk.speed >= self.cfg.hard_brake_min_speed_mps and
                    trk.acc < self.cfg.hard_brake_mps2):
                incidents.append({
                    "timestamp":        timestamp,
                    "frame":            frame_idx,
                    "type":             "Hard Braking",
                    "scenario":         "hard_brake",
                    "actor_1":          f"{trk.cls_name} (ID:{tid})",
                    "actor_2":          None,
                    "distance_m":       None,
                    "ttc_s":            None,
                    "speed_mps":        round(float(trk.speed), 2),
                    "acceleration_mps2": round(float(trk.acc), 2),
                    "risk":             "Medium",
                    "confidence":       round(float(trk.confidence), 3),
                })

            # Sudden swerve — update persist counter first, then fire once
            if self._is_swerve(trk):
                self.swerve_persist[tid] += 1
            else:
                self.swerve_persist[tid] = 0

            if self.swerve_persist[tid] == self.cfg.swerve_persist_frames:
                # Fire exactly once when the persist threshold is first reached
                self.swerve_persist[tid] = 0
                incidents.append({
                    "timestamp":         timestamp,
                    "frame":             frame_idx,
                    "type":              "Sudden Swerve",
                    "scenario":          "swerve",
                    "actor_1":           f"{trk.cls_name} (ID:{tid})",
                    "actor_2":           None,
                    "distance_m":        None,
                    "ttc_s":             None,
                    "speed_mps":         round(float(trk.speed), 2),
                    "lateral_acc_mps2":  round(float(trk.lateral_acc), 2),
                    "risk":              "Medium",
                    "confidence":        round(float(trk.confidence), 3),
                })

        return incidents