"""
Near-miss detection engine (all geometry in BEV metres).

Pair pipeline — gates in order:
   1. Cooldown            — pair recently fired
   2. Track age           — kinematics not yet stable
   3. Kinematics valid    — Kalman state + >=2 BEV positions
   4. Frame border        — homography unreliable at frame edges
   5. Coarse proximity    — centre-to-centre pre-filter
   6. Motion state        — two movers, or mover + stationary target
   7. Scenario + lateral  — path-frame lateral offset vs per-scenario limit
   8. Closing speed       — reject diverging pairs
   9. Track quality       — detection confidence + direction consistency
  10. CPA / TTC           — collision-course TTC against combined radii,
                            per-class-pair binary TTC gate
  11. Persistence         — leaky counter: soft gate failures decay by 1
                            instead of hard-resetting, so a single flickered
                            frame does not erase accumulated evidence
  12. Risk Index          — RI = α·R_d + β·R_ttc + γ·R_v

Key equations
-------------
Lateral path offset (per mover i, other at p_j):
    lat_i = | ĥ_i × (p_j − p_i) |        (2-D cross product, metres)
  This is the perpendicular offset of the other object from i's travel axis —
  a real, current-state quantity (unlike a predicted-drift hybrid).

Collision-course TTC (combined radius R = r1 + r2 + buffer):
    |dp + dv·t|² = R²  →  |dv|²·t² + 2(dp·dv)·t + |dp|² − R² = 0
  smallest non-negative root = time until footprints touch. Falls back to the
  closest-point-of-approach time when the trajectories never intersect R.

Motion intensity uses the RELATIVE velocity magnitude |v1 − v2| — the scalar
speed difference |s1 − s2| is 0 for two equal-speed vehicles meeting head-on,
which is exactly the case that must score highest.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from utils.motion import TrackState


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


class NearMissEngine:

    def __init__(self, cfg):
        self.cfg = cfg
        self.pair_persist: Dict[Tuple[int, int], int] = defaultdict(int)
        self.pair_cooldown: Dict[Tuple[int, int], int] = {}
        self.swerve_persist: Dict[int, int] = defaultdict(int)
        self.swerve_cooldown: Dict[int, int] = {}
        self.brake_persist: Dict[int, int] = defaultdict(int)
        self.brake_cooldown: Dict[int, int] = {}
        # Collision state machine (see _evaluate_collisions)
        self.pair_history: Dict[Tuple[int, int], deque] = {}   # pre-contact kinematics
        self.pair_contact: Dict[Tuple[int, int], dict] = {}    # active contact episodes
        self.collision_cooldown: Dict[Tuple[int, int], int] = {}

    # ── housekeeping ──────────────────────────────────────────────────────

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def prune(self, live_ids) -> None:
        """Drop per-pair / per-track state for dead tracks. Must be called
        periodically (each frame is fine) or state grows without bound on
        long-running streams."""
        live = set(live_ids)

        def _pair_ok(k):
            return k[0] in live and k[1] in live

        self.pair_persist = defaultdict(
            int, {k: v for k, v in self.pair_persist.items() if _pair_ok(k) and v > 0}
        )
        self.pair_cooldown = {k: v for k, v in self.pair_cooldown.items() if _pair_ok(k)}
        self.swerve_persist = defaultdict(
            int, {k: v for k, v in self.swerve_persist.items() if k in live and v > 0}
        )
        self.swerve_cooldown = {k: v for k, v in self.swerve_cooldown.items() if k in live}
        self.brake_persist = defaultdict(
            int, {k: v for k, v in self.brake_persist.items() if k in live and v > 0}
        )
        self.brake_cooldown = {k: v for k, v in self.brake_cooldown.items() if k in live}
        self.pair_history = {k: v for k, v in self.pair_history.items() if _pair_ok(k)}
        self.pair_contact = {k: v for k, v in self.pair_contact.items() if _pair_ok(k)}
        self.collision_cooldown = {
            k: v for k, v in self.collision_cooldown.items() if _pair_ok(k)
        }

    @staticmethod
    def _tick(cooldowns: dict) -> None:
        for k in list(cooldowns.keys()):
            cooldowns[k] -= 1
            if cooldowns[k] <= 0:
                del cooldowns[k]

    def _soft_fail(self, key) -> None:
        # Leaky persistence: decay instead of hard reset so one flickered
        # frame does not erase sustained evidence.
        if self.pair_persist[key] > 0:
            self.pair_persist[key] -= 1

    def _hard_fail(self, key) -> None:
        self.pair_persist[key] = 0

    # ── spatial helpers ───────────────────────────────────────────────────

    def _is_near_border(self, trk: TrackState) -> bool:
        if not trk.last_bbox:
            return False
        x1, y1, x2, y2 = trk.last_bbox
        m = self.cfg.frame_border_margin
        return (x1 < m or y1 < m or
                x2 > self.cfg.frame_width - m or
                y2 > self.cfg.frame_height - m)

    # ── scenario classification ───────────────────────────────────────────

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
            "rear_end":   self.cfg.lat_limit_rear_end,
            "merging":    self.cfg.lat_limit_merging,
            "crossing":   self.cfg.lat_limit_crossing,
            "head_on":    self.cfg.lat_limit_head_on,
            "stationary": self.cfg.lat_limit_stationary,
        }.get(scenario, self.cfg.lat_limit_crossing)

    # ── TTC / CPA ─────────────────────────────────────────────────────────

    @staticmethod
    def _conflict_times(
        dp: np.ndarray, dv: np.ndarray, radius: float, tmax: float
    ) -> Tuple[Optional[float], float, float]:
        """Solve the relative-motion conflict.

        dp = p2 − p1, dv = v2 − v1 (so relative separation(t) = dp + dv·t).
        Returns (t_hit, t_cpa, d_cpa):
          t_hit — smallest t in [0, tmax] with |dp + dv·t| = radius
                  (footprints touch), or None if no such time exists;
          t_cpa — time of closest approach, clipped to [0, tmax];
          d_cpa — separation at t_cpa.
        """
        dist = float(np.linalg.norm(dp))
        a = float(np.dot(dv, dv))
        if a < 1e-9:  # no relative motion
            return (0.0 if dist <= radius else None), 0.0, dist

        t_cpa = float(np.clip(-float(np.dot(dp, dv)) / a, 0.0, tmax))
        d_cpa = float(np.linalg.norm(dp + dv * t_cpa))

        c = dist * dist - radius * radius
        if c <= 0.0:
            return 0.0, t_cpa, d_cpa  # already overlapping
        b = 2.0 * float(np.dot(dp, dv))
        disc = b * b - 4.0 * a * c
        if disc <= 0.0:
            return None, t_cpa, d_cpa  # paths never come within `radius`
        t_hit = (-b - float(np.sqrt(disc))) / (2.0 * a)
        if 0.0 <= t_hit <= tmax:
            return t_hit, t_cpa, d_cpa
        return None, t_cpa, d_cpa

    def _get_ttc_threshold(self, cls1: str, cls2: str) -> float:
        key = f"{cls1}:{cls2}"
        rev = f"{cls2}:{cls1}"
        return float(
            self.cfg.ttc_threshold_by_pair.get(
                key,
                self.cfg.ttc_threshold_by_pair.get(rev, self.cfg.ttc_default_gate_s),
            )
        )

    # ── risk scoring ──────────────────────────────────────────────────────

    def _score_risk(
        self, edge_dist: float, ttc: float, rel_speed: float, vuln_heavy: bool
    ) -> Tuple[str, int, int, int, float]:
        """RI = α·R_d + β·R_ttc + γ·R_v (each component clipped to [0, 1])."""
        r_d = float(np.clip(
            1.0 - edge_dist / max(self.cfg.ri_distance_ref, 1e-6), 0.0, 1.0))
        r_ttc = float(np.clip(
            1.0 - ttc / max(self.cfg.ri_ttc_ref, 1e-6), 0.0, 1.0))
        r_v = float(np.clip(
            rel_speed / max(self.cfg.ri_v_max, 1e-6), 0.0, 1.0))

        ri = (self.cfg.ri_alpha * r_d +
              self.cfg.ri_beta * r_ttc +
              self.cfg.ri_gamma * r_v)
        if vuln_heavy:
            ri *= 1.10
        ri = float(np.clip(ri, 0.0, 1.0))

        if ri >= self.cfg.ri_high_threshold:
            risk = "High"
        elif ri >= self.cfg.ri_medium_threshold:
            risk = "Medium"
        else:
            risk = "Low"

        prob = int(np.clip(round(1 + 4 * max(r_d, r_ttc)), 1, 5))
        sev = int(np.clip(round(1 + 4 * r_v) + (1 if vuln_heavy else 0), 1, 5))
        score = int(np.clip(round(ri * 25), 1, 25))
        return risk, prob, sev, score, ri

    # ── swerve ────────────────────────────────────────────────────────────

    @staticmethod
    def _heading_change_deg(headings) -> float:
        if len(headings) < 2:
            return 0.0
        delta = headings[-1] - headings[0]
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360
        return abs(delta)

    def _is_swerve(self, trk: TrackState) -> bool:
        n = self.cfg.swerve_eval_frames
        if len(trk.lateral_acc_hist) < n or len(trk.heading_hist) < n:
            return False
        lat_vals = list(trk.lateral_acc_hist)[-n:]
        head_vals = list(trk.heading_hist)[-n:]
        return (
            trk.speed >= self.cfg.swerve_min_speed_mps
            and float(np.mean(lat_vals)) >= self.cfg.swerve_lateral_acc_threshold
            and float(np.max(lat_vals)) >= self.cfg.swerve_lateral_acc_threshold * 1.1
            and self._heading_change_deg(head_vals) >= self.cfg.swerve_heading_delta_min_deg
            and trk.direction_consistency >= self.cfg.direction_consistency_min
        )

    # ── collision detection ───────────────────────────────────────────────

    def _evaluate_collisions(
        self,
        frame_idx: int,
        timestamp: str,
        tracks: Dict[int, TrackState],
        video_time_s: Optional[float],
    ) -> List[dict]:
        """Per-pair contact state machine.

        A collision is confirmed only when three independent conditions hold:
          1. CONTACT  — edge distance <= collision_overlap_m for at least
                        collision_persist_frames;
          2. APPROACH — in the pre-contact window the pair was genuinely
                        converging (closing speed and relative speed above
                        thresholds). Adjacent-lane traffic whose circular
                        footprint approximations graze each other has
                        closing ≈ 0 and is rejected here;
          3. IMPACT   — within collision_post_window_frames after contact,
                        kinematic evidence of momentum exchange appears:
                        a deceleration spike on either track, a collapse of
                        relative speed, or both tracks coming to rest.
                        Occlusion pass-throughs (tracks crossing in image
                        space) separate at unchanged speed and never
                        produce this evidence.
        """
        cfg = self.cfg
        incidents: List[dict] = []
        ids = list(tracks.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                t1, t2 = tracks[id1], tracks[id2]
                key = self._pair_key(id1, id2)

                if key in self.collision_cooldown:
                    continue
                if (t1.age < cfg.collision_min_track_age or
                        t2.age < cfg.collision_min_track_age or
                        not (t1.kinematics_valid and t2.kinematics_valid)):
                    continue
                if self._is_near_border(t1) or self._is_near_border(t2):
                    continue

                p1 = np.asarray(t1.bev_positions[-1], dtype=np.float64)
                p2 = np.asarray(t2.bev_positions[-1], dtype=np.float64)
                dp = p2 - p1
                dist = float(np.linalg.norm(dp))

                if dist > cfg.collision_watch_radius_m:
                    self.pair_history.pop(key, None)
                    self.pair_contact.pop(key, None)
                    continue

                v1 = np.asarray(t1.vel, dtype=np.float64)
                v2 = np.asarray(t2.vel, dtype=np.float64)
                r1, r2 = t1.bev_radius_m, t2.bev_radius_m
                edge = max(0.0, dist - r1 - r2)
                u_sep = dp / max(dist, 1e-6)
                closing = float(np.dot(v1 - v2, u_sep))
                rel_speed = float(np.linalg.norm(v1 - v2))

                st = self.pair_contact.get(key)

                if st is None:
                    hist = self.pair_history.setdefault(
                        key, deque(maxlen=cfg.collision_pre_window_frames)
                    )
                    if edge <= cfg.collision_overlap_m and len(hist) >= 3:
                        # Condition 2 — approach evidence from the PRE-contact window
                        max_closing = max(h[0] for h in hist)
                        max_rel = max(h[1] for h in hist)
                        if (max_closing >= cfg.collision_min_closing_mps and
                                max_rel >= cfg.collision_min_impact_rel_speed_mps):
                            self.pair_contact[key] = {
                                "start": frame_idx,
                                "pre_rel": max_rel,
                                "pre_closing": max_closing,
                                "overlap_frames": 1,
                            }
                    hist.append((closing, rel_speed))
                    continue

                # Active contact episode — look for impact evidence (condition 3)
                if edge <= cfg.collision_overlap_m:
                    st["overlap_frames"] += 1

                decel_min = min(t1.acc, t2.acc)
                evidence_decel = decel_min <= cfg.collision_impact_decel_mps2
                evidence_rel_drop = (
                    rel_speed <= (1.0 - cfg.collision_rel_speed_drop_ratio) * st["pre_rel"]
                )
                evidence_stopped = (
                    max(t1.speed, t2.speed) <= cfg.collision_stopped_speed_mps
                )

                if (st["overlap_frames"] >= cfg.collision_persist_frames and
                        (evidence_decel or evidence_rel_drop or evidence_stopped)):
                    vuln = (t1.cls_name in cfg.vulnerable_classes or
                            t2.cls_name in cfg.vulnerable_classes)
                    incidents.append({
                        "timestamp":            timestamp,
                        "video_time_s":         round(video_time_s, 2) if video_time_s is not None else None,
                        "frame":                frame_idx,
                        "type":                 "Collision",
                        "scenario":             "collision",
                        "actor_1":              f"{t1.cls_name} (ID:{id1})",
                        "actor_2":              f"{t2.cls_name} (ID:{id2})",
                        "actor_1_id":           id1,
                        "actor_2_id":           id2,
                        "impact_rel_speed_mps": round(st["pre_rel"], 2),
                        "impact_rel_speed_kmh": round(st["pre_rel"] * 3.6, 1),
                        "post_rel_speed_mps":   round(rel_speed, 2),
                        "pre_closing_mps":      round(st["pre_closing"], 2),
                        "max_deceleration_mps2": round(decel_min, 2),
                        "overlap_frames":       st["overlap_frames"],
                        "edge_distance_m":      round(edge, 2),
                        "centre_distance_m":    round(dist, 2),
                        "evidence": [
                            e for e, ok in (
                                ("deceleration_spike", evidence_decel),
                                ("rel_speed_collapse", evidence_rel_drop),
                                ("both_stopped", evidence_stopped),
                            ) if ok
                        ],
                        "vulnerable_involved":  vuln,
                        "probability_level":    5,
                        "severity_level":       5,
                        "composite_score":      25,
                        "risk_index":           1.0,
                        "risk":                 "Critical",
                        "confidence":           round(float(min(t1.det_score, t2.det_score)), 3),
                    })
                    self.collision_cooldown[key] = cfg.collision_cooldown_frames
                    self.pair_contact.pop(key, None)
                    self.pair_history.pop(key, None)
                elif frame_idx - st["start"] > cfg.collision_post_window_frames:
                    # No impact signature in time — passing occlusion or a
                    # graze; abort and let the near-miss logic own this pair.
                    self.pair_contact.pop(key, None)
                    self.pair_history.pop(key, None)

        return incidents

    # ── main entry point ──────────────────────────────────────────────────

    def evaluate(
        self,
        frame_idx: int,
        timestamp: str,
        tracks: Dict[int, TrackState],
        video_time_s: Optional[float] = None,
    ) -> List[dict]:
        cfg = self.cfg
        incidents: List[dict] = []
        ids = list(tracks.keys())

        self._tick(self.pair_cooldown)
        self._tick(self.swerve_cooldown)
        self._tick(self.brake_cooldown)
        self._tick(self.collision_cooldown)

        # ── Collision detection (runs first; a confirmed collision
        #    silences near-miss alerts for that pair via its cooldown) ─────
        incidents.extend(
            self._evaluate_collisions(frame_idx, timestamp, tracks, video_time_s)
        )

        # ── Pairwise near-miss evaluation ─────────────────────────────────
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                t1, t2 = tracks[id1], tracks[id2]
                key = self._pair_key(id1, id2)

                # Gate 1 — cooldown (near-miss or confirmed-collision)
                if key in self.pair_cooldown or key in self.collision_cooldown:
                    continue

                # Gate 2 — track age
                if (t1.age < cfg.min_track_age_for_risk or
                        t2.age < cfg.min_track_age_for_risk):
                    self._hard_fail(key)
                    continue

                # Gate 3 — kinematics available
                if not (t1.kinematics_valid and t2.kinematics_valid):
                    self._hard_fail(key)
                    continue

                # Gate 4 — frame border
                if self._is_near_border(t1) or self._is_near_border(t2):
                    self._hard_fail(key)
                    continue

                p1 = np.asarray(t1.bev_positions[-1], dtype=np.float64)
                p2 = np.asarray(t2.bev_positions[-1], dtype=np.float64)
                v1 = np.asarray(t1.vel, dtype=np.float64)
                v2 = np.asarray(t2.vel, dtype=np.float64)
                s1, s2 = t1.speed, t2.speed

                # Gate 5 — coarse proximity
                dp = p2 - p1
                centre_dist = float(np.linalg.norm(dp))
                if centre_dist > cfg.proximity_gate_m:
                    self._soft_fail(key)
                    continue
                centre_dist = max(centre_dist, 1e-6)
                u_sep = dp / centre_dist

                # Gate 6 — motion state
                m1 = s1 >= cfg.min_speed_for_risk_mps
                m2 = s2 >= cfg.min_speed_for_risk_mps
                st1 = s1 < cfg.static_speed_mps
                st2 = s2 < cfg.static_speed_mps

                if m1 and m2:
                    pair_mode = "both_moving"
                elif (m1 and st2) or (m2 and st1):
                    pair_mode = "stationary_target"
                else:
                    # both slow, or one in the ambiguous speed band
                    self._soft_fail(key)
                    continue

                # Gate 7 — scenario + lateral path offset
                # lat_i = |ĥ_i × (p_j − p_i)|: perpendicular offset of the
                # other object from track i's travel axis.
                r1, r2 = t1.bev_radius_m, t2.bev_radius_m
                heading_delta: Optional[float] = None

                if pair_mode == "both_moving":
                    h1, h2 = v1 / s1, v2 / s2
                    cosang = float(np.clip(np.dot(h1, h2), -1.0, 1.0))
                    heading_delta = float(np.degrees(np.arccos(cosang)))
                    scenario = self._classify_scenario(heading_delta)

                    lat1 = abs(_cross2(h1, dp))
                    lat2 = abs(_cross2(h2, -dp))
                    lat_sep = min(lat1, lat2)
                    long_sep = abs(float(np.dot(dp, h1 if s1 >= s2 else h2)))
                else:
                    scenario = "stationary"
                    if m1:
                        h_m, p_rel = v1 / s1, dp        # target relative to mover
                    else:
                        h_m, p_rel = v2 / s2, -dp
                    long_m = float(np.dot(p_rel, h_m))
                    lat_sep = abs(_cross2(h_m, p_rel))
                    long_sep = abs(long_m)
                    # target must be ahead of the mover (allow slight overlap)
                    if long_m < -(r1 + r2):
                        self._soft_fail(key)
                        continue

                if lat_sep > self._lat_limit(scenario):
                    self._soft_fail(key)
                    continue

                # Gate 8 — closing speed along the separation axis
                closing = float(np.dot(v1 - v2, u_sep))
                if closing < cfg.min_closing_speed_mps:
                    self._soft_fail(key)
                    continue

                # Gate 9 — track quality (detection confidence is a real
                # detector score in [0,1]; direction consistency only for movers)
                if min(t1.det_score, t2.det_score) < cfg.min_confidence_for_risk:
                    self._soft_fail(key)
                    continue
                movers = [t for t, m in ((t1, m1), (t2, m2)) if m]
                if any(t.direction_consistency < cfg.direction_consistency_min
                       for t in movers):
                    self._soft_fail(key)
                    continue

                # Gate 10 — CPA / TTC
                radius = r1 + r2 + cfg.ttc_collision_buffer_m
                t_hit, t_cpa, d_cpa = self._conflict_times(
                    dp, v2 - v1, radius, cfg.ttc_max_eval_s
                )
                if d_cpa > cfg.cpa_miss_distance_m:
                    self._soft_fail(key)
                    continue

                edge_dist = max(0.0, centre_dist - r1 - r2)
                ttc = t_hit if t_hit is not None else t_cpa
                if scenario == "rear_end" and closing > 1e-6:
                    ttc = min(ttc, edge_dist / closing)  # 1-D range-rate TTC

                if ttc > self._get_ttc_threshold(t1.cls_name, t2.cls_name):
                    self._soft_fail(key)
                    continue

                # Gate 11 — persistence
                self.pair_persist[key] += 1
                if self.pair_persist[key] < cfg.incident_persist_frames:
                    continue

                # Gate 12 — Risk Index
                rel_speed = float(np.linalg.norm(v1 - v2))
                vuln_heavy = (
                    (t1.cls_name in cfg.vulnerable_classes and
                     t2.cls_name in cfg.heavy_classes) or
                    (t2.cls_name in cfg.vulnerable_classes and
                     t1.cls_name in cfg.heavy_classes)
                )
                risk, prob, sev, score, ri = self._score_risk(
                    edge_dist, ttc, rel_speed, vuln_heavy
                )

                incidents.append({
                    "timestamp":          timestamp,
                    "video_time_s":       round(video_time_s, 2) if video_time_s is not None else None,
                    "frame":              frame_idx,
                    "type":               f"Near-Miss ({scenario.replace('_', ' ').title()})",
                    "scenario":           scenario,
                    "actor_1":            f"{t1.cls_name} (ID:{id1})",
                    "actor_2":            f"{t2.cls_name} (ID:{id2})",
                    "actor_1_id":         id1,
                    "actor_2_id":         id2,
                    "edge_distance_m":    round(edge_dist, 2),
                    "centre_distance_m":  round(centre_dist, 2),
                    "ttc_s":              round(float(ttc), 2),
                    "cpa_distance_m":     round(d_cpa, 2),
                    "longitudinal_sep_m": round(long_sep, 2),
                    "lateral_sep_m":      round(lat_sep, 2),
                    "heading_delta_deg":  round(heading_delta, 1) if heading_delta is not None else None,
                    "closing_speed_mps":  round(closing, 2),
                    "relative_speed_mps": round(rel_speed, 2),
                    "relative_speed_kmh": round(rel_speed * 3.6, 1),
                    "probability_level":  prob,
                    "severity_level":     sev,
                    "composite_score":    score,
                    "risk_index":         round(ri, 3),
                    "risk":               risk,
                    "confidence":         round(float(min(t1.det_score, t2.det_score)), 3),
                })

                self.pair_cooldown[key] = cfg.incident_cooldown_frames
                self.pair_persist[key] = 0

        # ── Single-object events ──────────────────────────────────────────
        for tid, trk in tracks.items():
            if trk.age < cfg.min_track_age_for_risk or not trk.kinematics_valid:
                continue
            if self._is_near_border(trk):
                continue

            # Hard braking — persistence + cooldown so one manoeuvre fires once
            braking = (trk.speed >= cfg.hard_brake_min_speed_mps and
                       trk.acc <= cfg.hard_brake_mps2)
            if braking and tid not in self.brake_cooldown:
                self.brake_persist[tid] += 1
            else:
                self.brake_persist[tid] = 0

            if self.brake_persist[tid] >= cfg.hard_brake_persist_frames:
                self.brake_persist[tid] = 0
                self.brake_cooldown[tid] = cfg.hard_brake_cooldown_frames
                incidents.append({
                    "timestamp":         timestamp,
                    "video_time_s":      round(video_time_s, 2) if video_time_s is not None else None,
                    "frame":             frame_idx,
                    "type":              "Hard Braking",
                    "scenario":          "hard_brake",
                    "actor_1":           f"{trk.cls_name} (ID:{tid})",
                    "actor_1_id":        tid,
                    "actor_2":           None,
                    "speed_mps":         round(float(trk.speed), 2),
                    "acceleration_mps2": round(float(trk.acc), 2),
                    "risk":              "Medium",
                    "confidence":        round(float(trk.det_score), 3),
                })

            # Sudden swerve — persistence + cooldown
            if self._is_swerve(trk) and tid not in self.swerve_cooldown:
                self.swerve_persist[tid] += 1
            else:
                self.swerve_persist[tid] = 0

            if self.swerve_persist[tid] >= cfg.swerve_persist_frames:
                self.swerve_persist[tid] = 0
                self.swerve_cooldown[tid] = cfg.swerve_cooldown_frames
                incidents.append({
                    "timestamp":        timestamp,
                    "video_time_s":     round(video_time_s, 2) if video_time_s is not None else None,
                    "frame":            frame_idx,
                    "type":             "Sudden Swerve",
                    "scenario":         "swerve",
                    "actor_1":          f"{trk.cls_name} (ID:{tid})",
                    "actor_1_id":       tid,
                    "actor_2":          None,
                    "speed_mps":        round(float(trk.speed), 2),
                    "lateral_acc_mps2": round(float(trk.lateral_acc), 2),
                    "yaw_rate_dps":     round(float(np.degrees(trk.yaw_rate)), 1),
                    "risk":             "Medium",
                    "confidence":       round(float(trk.det_score), 3),
                })

        return incidents
