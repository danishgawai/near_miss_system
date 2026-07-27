"""
Near-miss / conflict engine — RITSMS-aligned logic on the local BEV pipeline.

Ported from the RITSMS reference service (nearmiss/utils/ttc_pet.py) and
adapted to run in-process on our TrackState objects (Kalman-filtered BEV
metres from utils.motion) instead of Kafka centroid messages.

Per frame it produces, for every road-user pair in spatiotemporal proximity:

  * TTC   — vectorised closed-form closest-point-of-approach over a
            constant-velocity model (combined-radius circle + buffer). Flags
            pairs actively closing on a collision course. Gated by an NFb3
            M-of-N temporal confirmer (last ~0.5 s, >= configured fraction).
  * PET   — spatial-grid arrival-gap detector. Flags pairs whose paths cross
            the same BEV cell at different times (crossing-course conflicts
            the TTC path structurally cannot see). Naturally delay-confirmed.
  * DRAC  — required closing-direction deceleration (reported, not a trigger).
  * Delta-V — perfectly-inelastic momentum severity proxy (reported).

TTC and PET are merged per pair: a pair may carry both, and the reported
level is the more severe of the two. Levels are SAFE / WARNING / CRITICAL;
only WARNING and CRITICAL are emitted. Metric outputs (DRAC, Delta-V) are
flagged conditional in BEV-only mode.

Kept beyond the RITSMS reference (value-adds): the actual-collision state
machine, hard-braking and sudden-swerve single-object events.
"""

import math
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Iterable, Set

from utils.motion import TrackState
from utils import measures as M


LEVEL_ORDER = {"SAFE": 0, "WARNING": 1, "CRITICAL": 2}


def _heading_delta_deg(h1: float, h2: float) -> float:
    return abs((h1 - h2 + 180.0) % 360.0 - 180.0)


# ─────────────────────────────────────────────────────────────────────────
# Vectorised closed-form pairwise TTC (upper-triangle only)
# ─────────────────────────────────────────────────────────────────────────
def compute_ttc_matrix(pos: np.ndarray, vel: np.ndarray, rad: np.ndarray,
                       buffer_m: float, horizon_s: float,
                       min_rel_v2: float, min_closing_mps: float) -> np.ndarray:
    """Symmetric n×n matrix of pairwise TTC (inf where no collision course).

    relative separation(t) = r + v·t, with r = p_j − p_i, v = v_j − v_i.
    A pair is a candidate only when its closing speed along the separation
    axis exceeds ``min_closing_mps`` — this rejects queued / crawling
    same-direction traffic whose footprint circles overlap without a genuine
    approach (the dominant false-positive source in dense scenes). The
    earliest t in [0, horizon] with |r + v·t| = R (R = r_i+r_j+buffer) is the
    TTC; already-overlapping genuinely-closing pairs get 0.
    """
    n = pos.shape[0]
    out = np.full((n, n), np.inf, dtype=np.float64)
    if n < 2:
        return out

    iu, ju = np.triu_indices(n, k=1)
    r = pos[ju] - pos[iu]
    v = vel[ju] - vel[iu]
    R = rad[iu] + rad[ju] + buffer_m

    r2 = np.einsum("ij,ij->i", r, r)
    v2 = np.einsum("ij,ij->i", v, v)
    rv = np.einsum("ij,ij->i", r, v)
    R2 = R * R

    # Closing speed = component of relative velocity along −r (positive while
    # the gap shrinks). Gate on its magnitude, not merely on the sign of r·v.
    dist = np.sqrt(np.maximum(r2, 1e-12))
    closing_speed = -rv / dist
    closing = closing_speed >= min_closing_mps

    pair = np.full(len(iu), np.inf, dtype=np.float64)
    in_radius = r2 <= R2
    pair[in_radius & closing] = 0.0

    solve = (~in_radius) & closing & (v2 > min_rel_v2)
    if solve.any():
        disc = rv[solve] * rv[solve] - v2[solve] * (r2[solve] - R2[solve])
        ok = disc >= 0.0
        if ok.any():
            sq = np.sqrt(disc[ok])
            v2v = v2[solve][ok]
            rvv = rv[solve][ok]
            t_enter = (-rvv - sq) / v2v
            in_h = (t_enter >= 0.0) & (t_enter <= horizon_s)
            idx = np.flatnonzero(solve)[ok][in_h]
            pair[idx] = t_enter[in_h]

    out[iu, ju] = pair
    out[ju, iu] = pair
    return out


# ─────────────────────────────────────────────────────────────────────────
# NFb3 temporal confirmer (TTC only) — M-of-N frames alerting.
# ─────────────────────────────────────────────────────────────────────────
class AlertConfirmer:
    def __init__(self, window_size: int, min_count: int):
        self._window = max(1, int(window_size))
        self._min = max(1, min(self._window, int(min_count)))
        self._history: Dict[Tuple[int, int], deque] = {}

    @property
    def window_size(self) -> int:
        return self._window

    @property
    def min_count(self) -> int:
        return self._min

    def update_frame(self, alerting_pairs: Iterable[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        now = set(alerting_pairs)
        confirmed: Set[Tuple[int, int]] = set()
        to_delete: List[Tuple[int, int]] = []
        for pair in set(self._history) | now:
            hist = self._history.setdefault(pair, deque(maxlen=self._window))
            alerting = pair in now
            hist.append(alerting)
            if alerting and sum(hist) >= self._min:
                confirmed.add(pair)
            elif len(hist) == self._window and not any(hist):
                to_delete.append(pair)
        for pair in to_delete:
            del self._history[pair]
        return confirmed

    def prune(self, live_pairs: Set[Tuple[int, int]]) -> None:
        for pair in [p for p in self._history if p not in live_pairs]:
            del self._history[pair]


# ─────────────────────────────────────────────────────────────────────────
# PET detector — spatial-grid Post-Encroachment Time.
# ─────────────────────────────────────────────────────────────────────────
class PETDetector:
    """Spatial-grid arrival-gap detector. Each cell keeps a time-ordered deque
    of visitor STAMPS (id + full kinematics), so a PET conflict can be resolved
    even after the earlier crosser has left the scene — the defining case for
    crossing conflicts (one user clears the point, another arrives later)."""

    def __init__(self, cell_size_m: float, max_window_s: float):
        self._cell = float(cell_size_m)
        self._max_window = float(max_window_s)
        self._grid: Dict[Tuple[int, int], deque] = {}

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self._cell)), int(math.floor(y / self._cell)))

    def visit_and_query(self, track_id: int, stamp: dict) -> Dict[int, Tuple[float, dict]]:
        """Append ``stamp`` (must include 't','x','y' + kinematics) to its cell
        and return {other_id: (pet, other_stamp)} — the smallest positive
        arrival gap per prior *other* track within the retention window."""
        t = stamp["t"]
        entries = self._grid.setdefault(self._key(stamp["x"], stamp["y"]), deque())
        cutoff = t - self._max_window
        while entries and entries[0]["t"] < cutoff:
            entries.popleft()

        smallest: Dict[int, Tuple[float, dict]] = {}
        for other in entries:
            oid = other["id"]
            if oid == track_id:
                continue
            pet = t - other["t"]
            if pet <= 0.0:
                continue
            prev = smallest.get(oid)
            if prev is None or pet < prev[0]:
                smallest[oid] = (pet, other)

        entries.append(stamp)
        return smallest

    def cleanup(self, now_s: float) -> None:
        cutoff = now_s - self._max_window
        empty = []
        for key, entries in self._grid.items():
            while entries and entries[0]["t"] < cutoff:
                entries.popleft()
            if not entries:
                empty.append(key)
        for k in empty:
            del self._grid[k]


class NearMissEngine:

    def __init__(self, cfg, fps: float = 30.0, site=None):
        self.cfg = cfg
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.site = site

        # NFb3 confirmation window derived from fps.
        window = max(1, int(round(self.fps * cfg.nfb3_window_s)))
        min_count = max(1, math.ceil(window * cfg.nfb3_valid_fraction))
        self.confirmer = AlertConfirmer(window, min_count)

        self.pet = PETDetector(cfg.pet_cell_size_m, cfg.pet_max_gap_s)

        self._min_level = LEVEL_ORDER.get(cfg.min_publish_level, 1)
        self._dedupe_frames = max(1, int(round(cfg.nearmiss_dedupe_s * self.fps)))

        # Per-pair state
        self.emit_cooldown: Dict[Tuple[int, int], int] = {}
        self.crit_streak: Dict[Tuple[int, int], int] = defaultdict(int)

        # Collision state machine
        self.pair_history: Dict[Tuple[int, int], deque] = {}
        self.pair_contact: Dict[Tuple[int, int], dict] = {}
        self.collision_cooldown: Dict[Tuple[int, int], int] = {}

        # Single-object events
        self.brake_persist: Dict[int, int] = defaultdict(int)
        self.brake_cooldown: Dict[int, int] = {}
        self.swerve_persist: Dict[int, int] = defaultdict(int)
        self.swerve_cooldown: Dict[int, int] = {}

    # ── housekeeping ──────────────────────────────────────────────────────

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _tick(cooldowns: dict) -> None:
        for k in list(cooldowns.keys()):
            cooldowns[k] -= 1
            if cooldowns[k] <= 0:
                del cooldowns[k]

    def prune(self, live_ids) -> None:
        live = set(live_ids)

        def _pair_ok(k):
            return k[0] in live and k[1] in live

        self.emit_cooldown = {k: v for k, v in self.emit_cooldown.items() if _pair_ok(k)}
        self.crit_streak = defaultdict(int, {k: v for k, v in self.crit_streak.items() if _pair_ok(k) and v > 0})
        self.pair_history = {k: v for k, v in self.pair_history.items() if _pair_ok(k)}
        self.pair_contact = {k: v for k, v in self.pair_contact.items() if _pair_ok(k)}
        self.collision_cooldown = {k: v for k, v in self.collision_cooldown.items() if _pair_ok(k)}
        self.brake_cooldown = {k: v for k, v in self.brake_cooldown.items() if k in live}
        self.swerve_cooldown = {k: v for k, v in self.swerve_cooldown.items() if k in live}
        self.brake_persist = defaultdict(int, {k: v for k, v in self.brake_persist.items() if k in live and v > 0})
        self.swerve_persist = defaultdict(int, {k: v for k, v in self.swerve_persist.items() if k in live and v > 0})

    # ── spatial / classification helpers ──────────────────────────────────

    def _is_near_border(self, trk: TrackState) -> bool:
        if not trk.last_bbox:
            return False
        x1, y1, x2, y2 = trk.last_bbox
        m = self.cfg.frame_border_margin
        return (x1 < m or y1 < m or
                x2 > self.cfg.frame_width - m or
                y2 > self.cfg.frame_height - m)

    def _in_roi(self, p) -> bool:
        return self.site is None or self.site.point_in_roi(p)

    def _ready(self, trk: TrackState, frame_idx: int) -> bool:
        """A track usable for conflict analysis this frame."""
        return (trk.age >= self.cfg.min_track_age_for_risk and
                trk.kinematics_valid and
                trk.det_score >= self.cfg.min_confidence_for_risk and
                (frame_idx - trk.last_update_frame) <= 1 and
                not self._is_near_border(trk) and
                self._in_roi(trk.bev_positions[-1]))

    def _scenario(self, angle_deg: Optional[float], pet: bool = False) -> str:
        if angle_deg is None:
            return "crossing"
        if pet:
            return "opposing_through" if angle_deg >= self.cfg.pet_opposing_min_deg else "crossing"
        if angle_deg <= self.cfg.heading_rear_end_max_deg:
            return "rear_end"
        if angle_deg <= self.cfg.heading_merging_max_deg:
            return "merging"
        if angle_deg <= self.cfg.heading_crossing_max_deg:
            return "crossing"
        return "head_on"

    def _ttc_level(self, ttc: float) -> str:
        if ttc < self.cfg.ttc_critical_s:
            return "CRITICAL"
        if ttc < self.cfg.ttc_screen_s:
            return "WARNING"
        return "SAFE"

    def _pet_level(self, pet: float, is_ped: bool) -> str:
        if pet < self.cfg.pet_critical_s:
            return "CRITICAL"
        gate = self.cfg.pet_gate_ped_s if is_ped else self.cfg.pet_gate_vv_s
        return "WARNING" if pet < gate else "SAFE"

    @staticmethod
    def _ped(cls_name: str) -> bool:
        return cls_name in ("pedestrian", "bicycle")

    def _level_score(self, level: str) -> float:
        return {"CRITICAL": 1.0, "WARNING": 0.6, "SAFE": 0.0}.get(level, 0.0)

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
        t = float(video_time_s) if video_time_s is not None else frame_idx / self.fps
        metric = bool(self.site and self.site.metric_confident)

        self._tick(self.emit_cooldown)
        self._tick(self.collision_cooldown)
        self._tick(self.brake_cooldown)
        self._tick(self.swerve_cooldown)

        # 1) Actual collisions (silence near-miss for that pair via cooldown).
        incidents.extend(self._evaluate_collisions(frame_idx, timestamp, tracks, video_time_s))

        # 2) Ready set for the conflict engine.
        ready: List[Tuple[int, TrackState]] = [
            (tid, trk) for tid, trk in tracks.items() if self._ready(trk, frame_idx)
        ]

        # 3) PET pass — stamp every ready track into the grid, collect gaps.
        # stamp_for caches per-id kinematics so an emit can reconstruct a pair
        # even when the earlier crosser has already left `tracks`.
        pet_raw: Dict[Tuple[int, int], dict] = {}
        stamp_for: Dict[int, dict] = {}
        for tid, trk in ready:
            p = trk.bev_positions[-1]
            v = trk.vel
            stamp = {"id": tid, "t": t, "x": float(p[0]), "y": float(p[1]),
                     "vx": float(v[0]), "vy": float(v[1]), "speed": float(trk.speed),
                     "cls": trk.cls_name, "rad": float(trk.bev_radius_m),
                     "det": float(trk.det_score)}
            stamp_for[tid] = stamp
            smallest = self.pet.visit_and_query(tid, stamp)
            for other_id, (pet_s, ostamp) in smallest.items():
                key = self._pair_key(tid, other_id)
                prev = pet_raw.get(key)
                if prev is None or pet_s < prev["pet"]:
                    pet_raw[key] = {"pet": pet_s, "arr": stamp, "early": ostamp}
        self.pet.cleanup(t)

        pet_results: Dict[Tuple[int, int], dict] = {}
        for key, d in pet_raw.items():
            a, b = d["arr"], d["early"]
            stamp_for.setdefault(a["id"], a)
            stamp_for.setdefault(b["id"], b)
            angle = math.degrees(M.interaction_angle((a["vx"], a["vy"]), (b["vx"], b["vy"])))
            if (a["speed"] < cfg.pet_direction_min_speed_mps or
                    b["speed"] < cfg.pet_direction_min_speed_mps or
                    angle < cfg.pet_cross_heading_min_deg):
                continue
            is_ped = self._ped(a["cls"]) or self._ped(b["cls"])
            lvl = self._pet_level(d["pet"], is_ped)
            if LEVEL_ORDER[lvl] >= self._min_level:
                pet_results[key] = {"pet": d["pet"], "level": lvl, "angle": angle}

        # 4) TTC pass — vectorised matrix over the ready set.
        ttc_results: Dict[Tuple[int, int], Tuple[float, str]] = {}
        alerting: Set[Tuple[int, int]] = set()
        n = len(ready)
        if n >= 2:
            pos = np.array([trk.bev_positions[-1] for _, trk in ready], dtype=np.float64)
            vel = np.array([trk.vel for _, trk in ready], dtype=np.float64)
            rad = np.array([trk.bev_radius_m for _, trk in ready], dtype=np.float64)
            ids = [tid for tid, _ in ready]

            mx = compute_ttc_matrix(pos, vel, rad, cfg.ttc_collision_buffer_m,
                                    cfg.ttc_horizon_s, cfg.ttc_min_rel_speed_mps ** 2,
                                    cfg.ttc_min_closing_mps)
            j_idx = np.argmin(mx, axis=1)
            candidates: Dict[Tuple[int, int], Tuple[float, str]] = {}
            for i in range(n):
                val = float(mx[i, j_idx[i]])
                if not math.isfinite(val):
                    continue
                lvl = self._ttc_level(val)
                if lvl == "SAFE":
                    continue
                key = self._pair_key(ids[i], ids[int(j_idx[i])])
                # Hysteresis: require sustained CRITICAL before escalating.
                if lvl == "CRITICAL":
                    self.crit_streak[key] += 1
                    if self.crit_streak[key] < cfg.ttc_hysteresis_frames:
                        lvl = "WARNING"
                else:
                    self.crit_streak[key] = 0
                alerting.add(key)
                prev = candidates.get(key)
                if prev is None or val < prev[0]:
                    candidates[key] = (val, lvl)

            confirmed = self.confirmer.update_frame(alerting)
            ttc_results = {k: candidates[k] for k in confirmed if k in candidates}
        else:
            self.confirmer.update_frame(set())

        # 5) Merge TTC + PET per pair; compute DRAC + Delta-V; emit WARNING+.
        def _info(idx):
            trk = tracks.get(idx)
            if trk is not None and trk.kinematics_valid:
                p = trk.bev_positions[-1]
                return {"pos": np.asarray(p, dtype=np.float64),
                        "vel": np.asarray(trk.vel, dtype=np.float64),
                        "speed": float(trk.speed), "cls": trk.cls_name,
                        "rad": float(trk.bev_radius_m), "det": float(trk.det_score)}
            s = stamp_for.get(idx)      # earlier crosser that has left the scene
            if s is None:
                return None
            return {"pos": np.array([s["x"], s["y"]], dtype=np.float64),
                    "vel": np.array([s["vx"], s["vy"]], dtype=np.float64),
                    "speed": s["speed"], "cls": s["cls"], "rad": s["rad"], "det": s["det"]}

        for key in set(ttc_results) | set(pet_results):
            if key in self.emit_cooldown or key in self.collision_cooldown:
                continue
            id_a, id_b = key
            ia, ib = _info(id_a), _info(id_b)
            if ia is None or ib is None:
                continue

            ttc_val, ttc_lvl = ttc_results.get(key, (None, "SAFE"))
            pet_entry = pet_results.get(key)
            pet_val, pet_lvl, angle = None, "SAFE", None
            if pet_entry is not None:
                pet_val, pet_lvl, angle = pet_entry["pet"], pet_entry["level"], pet_entry["angle"]

            final_lvl = ttc_lvl if LEVEL_ORDER[ttc_lvl] >= LEVEL_ORDER[pet_lvl] else pet_lvl
            if LEVEL_ORDER[final_lvl] < self._min_level:
                continue

            if angle is None:
                angle = math.degrees(M.interaction_angle(ia["vel"], ib["vel"]))

            if ttc_val is not None and pet_val is not None:
                kind = "TTC+PET"
            elif pet_val is not None:
                kind = "PET"
            else:
                kind = "TTC"
            scenario = self._scenario(angle, pet=(kind == "PET"))

            radius = ia["rad"] + ib["rad"] + cfg.ttc_collision_buffer_m
            drac = M.drac_2d(ia["pos"], ia["vel"], ib["pos"], ib["vel"], radius, cfg.ttc_closing_eps_m2s)
            theta = M.interaction_angle(ia["vel"], ib["vel"])
            m_a = cfg.class_mass_kg.get(ia["cls"], 2050.0)
            m_b = cfg.class_mass_kg.get(ib["cls"], 2050.0)
            dv = M.delta_v(ia["speed"], ib["speed"], m_a, m_b, theta)
            rel_speed = float(np.linalg.norm(ia["vel"] - ib["vel"]))
            r_sep = ib["pos"] - ia["pos"]
            d_sep = float(np.linalg.norm(r_sep))
            closing_speed = float(-np.dot(ib["vel"] - ia["vel"], r_sep) / d_sep) if d_sep > 1e-6 else 0.0
            conflict_pt = 0.5 * (ia["pos"] + ib["pos"])
            vuln = ia["cls"] in cfg.vulnerable_classes or ib["cls"] in cfg.vulnerable_classes

            incidents.append({
                "timestamp":            timestamp,
                "video_time_s":         round(t, 2),
                "frame":                frame_idx,
                "type":                 f"Near-Miss ({scenario.replace('_', ' ').title()})",
                "scenario":             scenario,
                "metric_kind":          kind,
                "actor_1":              f"{ia['cls']} (ID:{id_a})",
                "actor_2":              f"{ib['cls']} (ID:{id_b})",
                "actor_1_id":           id_a,
                "actor_2_id":           id_b,
                "level":                final_lvl,
                "ttc_s":                round(ttc_val, 2) if ttc_val is not None else None,
                "pet_s":                round(pet_val, 2) if pet_val is not None else None,
                "drac_mps2":            round(drac, 3) if drac is not None else None,
                "drac_conditional":     not metric,
                "deltav_mps":           round(dv, 2) if dv is not None else None,
                "deltav_conditional":   not metric,
                "relative_speed_mps":   round(rel_speed, 2),
                "closing_speed_mps":    round(closing_speed, 2),
                "interaction_angle_deg": round(float(angle), 1),
                "conflict_x_bev":       round(float(conflict_pt[0]), 2),
                "conflict_y_bev":       round(float(conflict_pt[1]), 2),
                "zones":                self.site.zones_containing(conflict_pt) if self.site else [],
                "vulnerable_involved":  vuln,
                "risk_index":           self._level_score(final_lvl),
                "confidence":           round(float(min(ia["det"], ib["det"])), 3),
            })
            self.emit_cooldown[key] = self._dedupe_frames

        # 6) Single-object events (kept beyond the reference).
        incidents.extend(self._single_object_events(frame_idx, timestamp, tracks, video_time_s))
        return incidents

    # ── single-object events (hard braking, sudden swerve) ────────────────

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
        cfg = self.cfg
        n = cfg.swerve_eval_frames
        if len(trk.lateral_acc_hist) < n or len(trk.heading_hist) < n:
            return False
        lat_vals = list(trk.lateral_acc_hist)[-n:]
        head_vals = list(trk.heading_hist)[-n:]
        return (
            trk.speed >= cfg.swerve_min_speed_mps
            and float(np.mean(lat_vals)) >= cfg.swerve_lateral_acc_threshold
            and float(np.max(lat_vals)) >= cfg.swerve_lateral_acc_threshold * 1.1
            and self._heading_change_deg(head_vals) >= cfg.swerve_heading_delta_min_deg
            and trk.direction_consistency >= cfg.direction_consistency_min
        )

    def _single_object_events(self, frame_idx, timestamp, tracks, video_time_s) -> List[dict]:
        cfg = self.cfg
        out: List[dict] = []
        vt = round(video_time_s, 2) if video_time_s is not None else None
        for tid, trk in tracks.items():
            if trk.age < cfg.min_track_age_for_risk or not trk.kinematics_valid:
                continue
            if self._is_near_border(trk):
                continue

            braking = (trk.speed >= cfg.hard_brake_min_speed_mps and trk.acc <= cfg.hard_brake_mps2)
            if braking and tid not in self.brake_cooldown:
                self.brake_persist[tid] += 1
            else:
                self.brake_persist[tid] = 0
            if self.brake_persist[tid] >= cfg.hard_brake_persist_frames:
                self.brake_persist[tid] = 0
                self.brake_cooldown[tid] = cfg.hard_brake_cooldown_frames
                out.append({
                    "timestamp": timestamp, "video_time_s": vt, "frame": frame_idx,
                    "type": "Hard Braking", "scenario": "hard_brake",
                    "actor_1": f"{trk.cls_name} (ID:{tid})", "actor_1_id": tid, "actor_2": None,
                    "speed_mps": round(float(trk.speed), 2),
                    "acceleration_mps2": round(float(trk.acc), 2),
                    "level": "WARNING", "risk_index": 0.6,
                    "confidence": round(float(trk.det_score), 3),
                })

            if self._is_swerve(trk) and tid not in self.swerve_cooldown:
                self.swerve_persist[tid] += 1
            else:
                self.swerve_persist[tid] = 0
            if self.swerve_persist[tid] >= cfg.swerve_persist_frames:
                self.swerve_persist[tid] = 0
                self.swerve_cooldown[tid] = cfg.swerve_cooldown_frames
                out.append({
                    "timestamp": timestamp, "video_time_s": vt, "frame": frame_idx,
                    "type": "Sudden Swerve", "scenario": "swerve",
                    "actor_1": f"{trk.cls_name} (ID:{tid})", "actor_1_id": tid, "actor_2": None,
                    "speed_mps": round(float(trk.speed), 2),
                    "lateral_acc_mps2": round(float(trk.lateral_acc), 2),
                    "yaw_rate_dps": round(float(np.degrees(trk.yaw_rate)), 1),
                    "level": "WARNING", "risk_index": 0.6,
                    "confidence": round(float(trk.det_score), 3),
                })
        return out

    # ── actual-collision state machine (value-add beyond RITSMS) ──────────

    def _evaluate_collisions(self, frame_idx, timestamp, tracks, video_time_s) -> List[dict]:
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
                        key, deque(maxlen=cfg.collision_pre_window_frames))
                    if edge <= cfg.collision_overlap_m and len(hist) >= 3:
                        max_closing = max(h[0] for h in hist)
                        max_rel = max(h[1] for h in hist)
                        if (max_closing >= cfg.collision_min_closing_mps and
                                max_rel >= cfg.collision_min_impact_rel_speed_mps):
                            self.pair_contact[key] = {
                                "start": frame_idx, "pre_rel": max_rel,
                                "pre_closing": max_closing, "overlap_frames": 1,
                            }
                    hist.append((closing, rel_speed))
                    continue

                if edge <= cfg.collision_overlap_m:
                    st["overlap_frames"] += 1

                decel_min = min(t1.acc, t2.acc)
                evidence_decel = decel_min <= cfg.collision_impact_decel_mps2
                evidence_rel_drop = rel_speed <= (1.0 - cfg.collision_rel_speed_drop_ratio) * st["pre_rel"]
                evidence_stopped = max(t1.speed, t2.speed) <= cfg.collision_stopped_speed_mps

                if (st["overlap_frames"] >= cfg.collision_persist_frames and
                        (evidence_decel or evidence_rel_drop or evidence_stopped)):
                    vuln = (t1.cls_name in cfg.vulnerable_classes or
                            t2.cls_name in cfg.vulnerable_classes)
                    incidents.append({
                        "timestamp": timestamp,
                        "video_time_s": round(video_time_s, 2) if video_time_s is not None else None,
                        "frame": frame_idx,
                        "type": "Collision", "scenario": "collision",
                        "actor_1": f"{t1.cls_name} (ID:{id1})",
                        "actor_2": f"{t2.cls_name} (ID:{id2})",
                        "actor_1_id": id1, "actor_2_id": id2,
                        "metric_kind": "COLLISION",
                        "impact_rel_speed_mps": round(st["pre_rel"], 2),
                        "impact_rel_speed_kmh": round(st["pre_rel"] * 3.6, 1),
                        "post_rel_speed_mps": round(rel_speed, 2),
                        "pre_closing_mps": round(st["pre_closing"], 2),
                        "max_deceleration_mps2": round(decel_min, 2),
                        "overlap_frames": st["overlap_frames"],
                        "edge_distance_m": round(edge, 2),
                        "centre_distance_m": round(dist, 2),
                        "conflict_x_bev": round(float(0.5 * (p1[0] + p2[0])), 2),
                        "conflict_y_bev": round(float(0.5 * (p1[1] + p2[1])), 2),
                        "evidence": [e for e, ok in (
                            ("deceleration_spike", evidence_decel),
                            ("rel_speed_collapse", evidence_rel_drop),
                            ("both_stopped", evidence_stopped),
                        ) if ok],
                        "vulnerable_involved": vuln,
                        "level": "CRITICAL", "risk_index": 1.0,
                        "confidence": round(float(min(t1.det_score, t2.det_score)), 3),
                    })
                    self.collision_cooldown[key] = cfg.collision_cooldown_frames
                    self.pair_contact.pop(key, None)
                    self.pair_history.pop(key, None)
                elif frame_idx - st["start"] > cfg.collision_post_window_frames:
                    self.pair_contact.pop(key, None)
                    self.pair_history.pop(key, None)

        return incidents
