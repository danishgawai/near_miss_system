"""§4.5 — Conflict extraction (forecast-driven).

For every spatiotemporally-proximate pair the engine computes the proposal's
measures, using the per-track forecasts (CV/CTRV) from trajectory.py:

  * TTC  — earliest time the two PREDICTED oriented footprints overlap
           (separating-axis test sampled along the forecasts). Because the
           forecast follows the curve for turning vehicles, this is correct for
           turning/angled approaches, not just straight-line closing. Gated by
           a minimum closing speed (rejects queued traffic) and confirmed by
           the NFb3 rule (>= fraction of the last 0.5 s below the 3 s window).
  * PET  — hybrid: observational arrival-gap on a BEV grid (a departed crosser
           still counts, via stored stamps), PREDICTIVE arrival-gap where the
           two forecasts cross a common point, and zone entry/exit gap where
           conflict zones are configured. The smallest valid PET governs.
  * DRAC, Delta-V — reported per pair (conditional on metric scale).

Levels are SAFE / WARNING / CRITICAL; only WARNING/CRITICAL are emitted, one
record per pair per dedupe window.
"""

import math
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from utils import measures as M
from ritsms.patterns import crash_codes_for

LEVEL_ORDER = {"SAFE": 0, "WARNING": 1, "CRITICAL": 2}


def _rel_angle_deg(h1: float, h2: float) -> float:
    return abs((h1 - h2 + 180.0) % 360.0 - 180.0)


# ── predictive geometry over forecasts ────────────────────────────────────

def predictive_ttc(fa, fb, la, wa, lb, wb, buffer_m) -> Optional[float]:
    """Earliest forecast time the two oriented footprints overlap (SAT).
    fa/fb: [t,x,y,theta] arrays on the same time grid."""
    n = min(len(fa), len(fb))
    La, Wa = la + buffer_m, wa + buffer_m
    Lb, Wb = lb + buffer_m, wb + buffer_m
    for k in range(n):
        A = M.rect_corners((fa[k, 1], fa[k, 2]), fa[k, 3], La, Wa)
        B = M.rect_corners((fb[k, 1], fb[k, 2]), fb[k, 3], Lb, Wb)
        if M._obb_overlap(A, B):
            return float(fa[k, 0])
    return None


def _seg_intersect(p1, p2, p3, p4):
    """Fractional intersection (t on p1->p2, u on p3->p4) of two segments, or
    None if they are parallel/collinear or do not cross within their extents."""
    r = p2 - p1
    s = p4 - p3
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-9:
        return None
    qp = p3 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u
    return None


def predictive_pet(fa, fb, step=3, min_future_s=0.05) -> Optional[Tuple[float, np.ndarray]]:
    """Predictive PET from a genuine FUTURE path crossing.

    Finds where the two forecast polylines actually intersect in space (not
    merely come close — a pair that is close *now* is a TTC case, not a PET
    one), and returns |arrival-time difference| at the earliest such crossing
    that lies in the future for BOTH tracks. None if the paths never cross.
    """
    A = fa[::step]
    B = fb[::step]
    best = None
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            hit = _seg_intersect(A[i, 1:3], A[i + 1, 1:3], B[j, 1:3], B[j + 1, 1:3])
            if hit is None:
                continue
            ta = A[i, 0] + (A[i + 1, 0] - A[i, 0]) * hit[0]
            tb = B[j, 0] + (B[j + 1, 0] - B[j, 0]) * hit[1]
            if ta <= min_future_s or tb <= min_future_s:
                continue            # crossing is now/behind -> not a PET gap
            pet = abs(ta - tb)
            pt = A[i, 1:3] + (A[i + 1, 1:3] - A[i, 1:3]) * hit[0]
            key = max(ta, tb)       # earliest fully-realised crossing
            if best is None or key < best[0]:
                best = (key, pet, pt)
    if best is None:
        return None
    return best[1], best[2]


# ── observational PET grid (arrival gap; departed crosser retained) ───────

class PETGrid:
    def __init__(self, cell_size_m, max_window_s):
        self._cell = float(cell_size_m)
        self._win = float(max_window_s)
        self._grid: Dict[Tuple[int, int], deque] = {}

    def _key(self, x, y):
        return (int(math.floor(x / self._cell)), int(math.floor(y / self._cell)))

    def visit(self, stamp) -> Dict[int, Tuple[float, dict]]:
        t = stamp["t"]
        cell = self._grid.setdefault(self._key(stamp["x"], stamp["y"]), deque())
        while cell and cell[0]["t"] < t - self._win:
            cell.popleft()
        out: Dict[int, Tuple[float, dict]] = {}
        for other in cell:
            if other["id"] == stamp["id"]:
                continue
            pet = t - other["t"]
            if pet <= 0:
                continue
            if other["id"] not in out or pet < out[other["id"]][0]:
                out[other["id"]] = (pet, other)
        cell.append(stamp)
        return out

    def cleanup(self, now):
        for k in [k for k, c in self._grid.items()
                  if not c or c[-1]["t"] < now - self._win]:
            self._grid.pop(k, None)


# ── zone entry/exit PET (proposal §4.5 conflict zones) ────────────────────

class ZoneTracker:
    def __init__(self, max_window_s):
        self._win = float(max_window_s)
        self._occ: Dict[Tuple[str, int], dict] = {}   # (zone, tid) -> {enter,last}

    def update(self, tid, zones_now, t):
        for z in zones_now:
            rec = self._occ.get((z, tid))
            if rec is None:
                self._occ[(z, tid)] = {"enter": t, "last": t}
            else:
                rec["last"] = t

    def pet_for_pair(self, a_id, b_id, t) -> Optional[Tuple[float, str]]:
        best = None
        for (z, tid), rec in list(self._occ.items()):
            if t - rec["last"] > self._win:
                self._occ.pop((z, tid), None)
        za = {z: r for (z, tid), r in self._occ.items() if tid == a_id}
        zb = {z: r for (z, tid), r in self._occ.items() if tid == b_id}
        for z in set(za) & set(zb):
            ra, rb = za[z], zb[z]
            # later arriver's enter minus earlier user's exit(last-seen)
            if ra["enter"] <= rb["enter"]:
                pet = rb["enter"] - ra["last"]
            else:
                pet = ra["enter"] - rb["last"]
            pet = abs(pet)
            if best is None or pet < best[0]:
                best = (pet, z)
        return best


# ── NFb3 confirmer (TTC only) ─────────────────────────────────────────────

class NFb3Confirmer:
    def __init__(self, window, min_count):
        self._w = max(1, int(window))
        self._m = max(1, min(self._w, int(min_count)))
        self._hist: Dict[Tuple[int, int], deque] = {}

    def update(self, pair, below_threshold: bool) -> Tuple[int, int, bool]:
        h = self._hist.setdefault(pair, deque(maxlen=self._w))
        h.append(bool(below_threshold))
        nfb3 = sum(h)
        return nfb3, len(h), (nfb3 >= self._m)

    def prune(self, live_pairs):
        for p in [p for p in self._hist if p not in live_pairs]:
            self._hist.pop(p, None)


class ConflictEngine:
    def __init__(self, cfg, fps, site=None):
        self.cfg = cfg
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.site = site
        self.pet_grid = PETGrid(cfg.pet_cell_size_m, cfg.pet_max_gap_s)
        self.zones = ZoneTracker(cfg.pet_max_gap_s)
        window = max(1, int(round(self.fps * cfg.nfb3_window_s)))
        self.nfb3 = NFb3Confirmer(window, math.ceil(window * cfg.nfb3_valid_fraction))
        self.emit_cooldown: Dict[Tuple[int, int], int] = {}
        self._dedupe = max(1, int(round(cfg.nearmiss_dedupe_s * self.fps)))
        self._conflict_seq = 0
        self.frame_levels: Dict[int, str] = {}   # live per-track level this frame
        self.frame_active: List[dict] = []       # live WARNING+ pairs this frame
        self.frame_traces: List[dict] = []       # per-pair measure traces this frame

    # -- helpers --
    @staticmethod
    def _pair(a, b):
        return (a, b) if a < b else (b, a)

    def _is_border(self, trk):
        if not trk.last_bbox:
            return False
        x1, y1, x2, y2 = trk.last_bbox
        m = self.cfg.frame_border_margin
        return (x1 < m or y1 < m or x2 > self.cfg.frame_width - m or y2 > self.cfg.frame_height - m)

    def _ready(self, trk, frame_idx):
        return (trk.ready and trk.age >= self.cfg.min_track_age and
                trk.det_score >= self.cfg.min_confidence and
                (frame_idx - trk.last_update_frame) <= 1 and
                not self._is_border(trk) and
                (self.site is None or self.site.point_in_roi(trk.position_m)))

    def _ped(self, cls):
        return cls in self.cfg.ped_classes

    def _encounter(self, a, b, angle_deg):
        cfg = self.cfg
        vuln = a.cls_name in cfg.vulnerable_classes or b.cls_name in cfg.vulnerable_classes
        turning = max(abs(math.degrees(a.yaw_rate)), abs(math.degrees(b.yaw_rate))) >= cfg.ctrv_min_yaw_rate_dps
        if angle_deg <= cfg.heading_rear_end_max_deg:
            base, gov = "rear_end", "TTC"
        elif angle_deg <= cfg.heading_merging_max_deg:
            base, gov = ("turning" if turning else "lane_change_merge"), "TTC"
        elif angle_deg <= cfg.heading_crossing_max_deg:
            base, gov = "crossing", "PET"
        else:
            base, gov = ("opposing_through" if self._ped(a.cls_name) or self._ped(b.cls_name) or turning else "head_on"), \
                        ("PET" if turning else "TTC")
        if vuln:
            base = f"vehicle_vru_{base}"
            gov = "PET" if base.endswith(("crossing", "opposing_through")) else gov
        return base, gov

    def _ttc_level(self, ttc):
        if ttc < self.cfg.ttc_critical_s:
            return "CRITICAL"
        if ttc < self.cfg.ttc_warning_s:
            return "WARNING"
        return "SAFE"

    def _pet_level(self, pet, ped):
        if pet < self.cfg.pet_critical_s:
            return "CRITICAL"
        gate = self.cfg.pet_warning_ped_s if ped else self.cfg.pet_warning_vv_s
        return "WARNING" if pet < gate else "SAFE"

    @staticmethod
    def _tick(cd):
        for k in list(cd):
            cd[k] -= 1
            if cd[k] <= 0:
                del cd[k]

    # -- main --
    def evaluate(self, frame_idx, timestamp, correct_ts, tracks: dict, t_s) -> List[dict]:
        cfg = self.cfg
        self._tick(self.emit_cooldown)
        metric = bool(self.site and self.site.metric_confident)

        # Live per-frame status (NOT dedupe-gated) — drives box highlighting so a
        # track stays coloured for the whole time its pair is WARNING/CRITICAL.
        frame_levels: Dict[int, str] = {}
        frame_active: List[dict] = []
        # Per-pair measure traces (every proximate pair with a computed measure)
        # for AC6 TTC-vs-frame validation plots.
        frame_traces: List[dict] = []

        ready = [(tid, trk) for tid, trk in tracks.items() if self._ready(trk, frame_idx)]
        live_pairs = set()

        # forecasts + PET-grid stamps + zone occupancy (one pass)
        fc: Dict[int, np.ndarray] = {}
        pet_obs: Dict[Tuple[int, int], Tuple[float, dict, int]] = {}
        stamp_for: Dict[int, dict] = {}
        for tid, trk in ready:
            fc[tid] = trk.forecast()
            p = trk.position_m
            stamp = {"id": tid, "t": t_s, "x": float(p[0]), "y": float(p[1]),
                     "vx": float(trk.velocity[0]), "vy": float(trk.velocity[1]),
                     "speed": trk.speed, "cls": trk.cls_name, "heading": trk.heading_deg}
            stamp_for[tid] = stamp
            if self.site is not None and self.site.zones:
                self.zones.update(tid, self.site.zones_containing(p), t_s)
            for oid, (pet_s, ostamp) in self.pet_grid.visit(stamp).items():
                key = self._pair(tid, oid)
                if key not in pet_obs or pet_s < pet_obs[key][0]:
                    pet_obs[key] = (pet_s, ostamp, tid)
                stamp_for.setdefault(oid, ostamp)
        self.pet_grid.cleanup(t_s)

        incidents: List[dict] = []
        n = len(ready)
        for i in range(n):
            for j in range(i + 1, n):
                id_a, trk_a = ready[i]
                id_b, trk_b = ready[j]
                pa, pb = trk_a.position_m, trk_b.position_m
                centre = float(np.linalg.norm(pa - pb))
                if centre > cfg.proximity_gate_m:
                    continue
                key = self._pair(id_a, id_b)
                live_pairs.add(key)

                va, vb = trk_a.velocity, trk_b.velocity
                r = pb - pa
                dist = max(centre, 1e-6)
                closing = float(np.dot(va - vb, r / dist))

                # ---- motion state (measures doc: an interaction is a pair of
                # MOVING objects). A vehicle stopped at a signal is a legitimate
                # conflict TARGET (rear-end into a queue is a real, common
                # conflict) but not an initiator, and at ~0 speed its velocity
                # direction is pure noise — which fabricated "head_on" events at
                # IP33B against vehicles stopped on the OPPOSITE carriageway.
                a_moving = trk_a.speed >= cfg.min_moving_speed_mps
                b_moving = trk_b.speed >= cfg.min_moving_speed_mps
                if not (a_moving or b_moving):
                    continue                      # both stopped — no interaction
                stationary_target = not (a_moving and b_moving)
                if stationary_target:
                    # The mover must genuinely be on course to hit the stopped
                    # vehicle: ahead of it and within a lateral corridor. This is
                    # what separates "approaching the back of my own queue"
                    # (a real rear-end) from "driving past a queue stopped on the
                    # opposite lane" (not a conflict).
                    if a_moving:
                        mover, target = trk_a, trk_b
                        mv, mp, tp = va, pa, pb
                    else:
                        mover, target = trk_b, trk_a
                        mv, mp, tp = vb, pb, pa
                    sp = float(np.linalg.norm(mv))
                    if sp < 1e-6:
                        continue
                    h = mv / sp
                    rel = tp - mp
                    lon = float(np.dot(rel, h))
                    lat = abs(float(rel[0] * h[1] - rel[1] * h[0]))
                    if lon <= 0.0 or lat > cfg.stationary_lat_corridor_m:
                        continue
                    # Classify using the stopped vehicle's heading from BEFORE it
                    # stopped (it kept the direction it approached the signal in),
                    # never its ~0-speed noise direction. Falls back to the
                    # mover's own axis when no moving heading was ever observed.
                    tgt_h = target.last_moving_heading_deg
                    if tgt_h is None:
                        angle = 0.0
                    else:
                        angle = _rel_angle_deg(math.degrees(math.atan2(h[1], h[0])), tgt_h)
                else:
                    angle = math.degrees(M.interaction_angle(va, vb))

                # ---- lane discipline for SAME-DIRECTION pairs ----------------
                # The rear-end TTC formula assumes both users are "on the same
                # line of travel". An adjacent-lane pair breaks that premise, so
                # it is either a lane-change/side-swipe (if actually converging
                # laterally) or simple parallel travel (no conflict at all).
                same_dir_override = None
                if angle <= cfg.heading_rear_end_max_deg:
                    fast, slow = ((trk_a, trk_b) if trk_a.speed >= trk_b.speed
                                  else (trk_b, trk_a))
                    sp = float(np.linalg.norm(fast.velocity))
                    if sp > 1e-6:
                        h = np.asarray(fast.velocity, dtype=np.float64) / sp
                        rel = np.asarray(slow.position_m, dtype=np.float64) - \
                              np.asarray(fast.position_m, dtype=np.float64)
                        lat = abs(float(rel[0] * h[1] - rel[1] * h[0]))
                        if lat > cfg.lat_limit_rear_end_m:
                            # lateral closing rate: is the gap across lanes shrinking?
                            dv = np.asarray(slow.velocity, dtype=np.float64) - \
                                 np.asarray(fast.velocity, dtype=np.float64)
                            lat_rate = float(dv[0] * h[1] - dv[1] * h[0])
                            side = float(rel[0] * h[1] - rel[1] * h[0])
                            lat_closing = -lat_rate * np.sign(side) if side != 0 else 0.0
                            if lat_closing < cfg.sideswipe_min_lat_closing_mps:
                                continue          # parallel travel in separate lanes
                            same_dir_override = "lane_change_merge"

                encounter, governing = self._encounter(trk_a, trk_b, angle)
                if same_dir_override:
                    vuln = (trk_a.cls_name in cfg.vulnerable_classes or
                            trk_b.cls_name in cfg.vulnerable_classes)
                    encounter = (f"vehicle_vru_{same_dir_override}" if vuln
                                 else same_dir_override)
                    governing = "TTC"
                ped = self._ped(trk_a.cls_name) or self._ped(trk_b.cls_name)

                ttc, ttc_lvl, pet, pet_lvl, pet_src = None, "SAFE", None, "SAFE", None
                nfb3, nwin = 0, 0

                if governing == "TTC":
                    # Coinciding conflict (rear-end / head-on / merge / sideswipe):
                    # earliest predicted footprint overlap, closing-gated, NFb3-confirmed.
                    if closing >= cfg.ttc_min_closing_mps:
                        ttc = predictive_ttc(fc[id_a], fc[id_b],
                                             trk_a.length_m, trk_a.width_m,
                                             trk_b.length_m, trk_b.width_m,
                                             cfg.footprint_buffer_m)
                    # ttc == 0 means the footprints ALREADY overlap, i.e. contact
                    # has occurred. For road users that is a collision, not a
                    # near-miss, and in practice it is nearly always a modelling
                    # artifact (footprint priors vs occlusion-corrupted reference
                    # points). Require a genuine future collision course.
                    if ttc is not None and ttc <= cfg.ttc_min_reportable_s:
                        ttc = None
                    below = ttc is not None and ttc < cfg.ttc_warning_s
                    nfb3, nwin, nfb3_ok = self.nfb3.update(key, below)
                    if ttc is not None and nfb3_ok:
                        ttc_lvl = self._ttc_level(ttc)
                    final = ttc_lvl
                else:
                    # Crossing conflict (crossing / opposing-through / VRU):
                    # hybrid PET — observational grid, predictive forecast-crossing,
                    # and zone entry/exit; the smallest valid gap governs.
                    cands = []
                    if key in pet_obs:
                        cands.append((pet_obs[key][0], "observed"))
                    pp = predictive_pet(fc[id_a], fc[id_b])
                    if pp is not None:
                        cands.append((pp[0], "predicted"))
                    zp = self.zones.pet_for_pair(id_a, id_b, t_s) if (self.site and self.site.zones) else None
                    if zp is not None:
                        cands.append((zp[0], f"zone:{zp[1]}"))
                    # Discard sub-floor gaps: those are simultaneous occupation,
                    # not an encroachment gap (see cfg.pet_min_s).
                    cands = [c for c in cands if c[0] >= cfg.pet_min_s]
                    if cands:
                        pet, pet_src = min(cands, key=lambda c: c[0])
                        pet_lvl = self._pet_level(pet, ped)
                    final = pet_lvl

                # Trace every proximate pair that has a computed measure (for the
                # AC6 TTC-vs-frame plots) — regardless of level/dedupe.
                if ttc is not None or pet is not None:
                    frame_traces.append({
                        "frame": frame_idx, "t_s": round(t_s, 3),
                        "id1": id_a, "id2": id_b,
                        "ttc": round(ttc, 3) if ttc is not None else None,
                        "pet": round(pet, 3) if pet is not None else None,
                        "level": final, "encounter": encounter,
                    })

                if LEVEL_ORDER[final] < LEVEL_ORDER.get(cfg.min_publish_level, 1):
                    continue

                # Live status: mark both tracks at this level every frame the
                # condition holds (independent of the record-dedupe below).
                frame_active.append({
                    "roaduser1_id": id_a, "roaduser2_id": id_b, "level": final,
                    "encounter_type": encounter,
                    "ttc_s": round(ttc, 2) if ttc is not None else None,
                    "pet_s": round(pet, 2) if pet is not None else None,
                })
                for _tid in (id_a, id_b):
                    prev = frame_levels.get(_tid)
                    if prev is None or LEVEL_ORDER[final] > LEVEL_ORDER[prev]:
                        frame_levels[_tid] = final

                # Record emission is deduped (one row per pair per window) for
                # the §7.2 conflict file / counts.
                if key in self.emit_cooldown:
                    continue

                # ---- reported measures ----
                radius = trk_a.radius_m + trk_b.radius_m + cfg.footprint_buffer_m
                drac = M.drac_2d(pa, va, pb, vb, radius, 0.1)
                theta = M.interaction_angle(va, vb)
                dv = M.delta_v(trk_a.speed, trk_b.speed,
                               cfg.class_mass_kg.get(trk_a.cls_name, 2050.0),
                               cfg.class_mass_kg.get(trk_b.cls_name, 2050.0), theta)
                cpt = 0.5 * (pa + pb)
                zones = self.site.zones_containing(cpt) if self.site else []
                self._conflict_seq += 1

                # Georeference the conflict point when the site is metric-validated
                # (lat/lng GCP calibration) -- BEV metres are true East/North then.
                geo = getattr(self.site, "geo", None) if self.site else None
                conflict_geo = geo.geo_for(float(cpt[0]), float(cpt[1])) if geo else None

                incidents.append({
                    "site_id": cfg.site_id,
                    "conflict_id": self._conflict_seq,
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "correct_timestamp": correct_ts,
                    "video_time_s": round(t_s, 3),
                    "roaduser1_id": id_a, "roaduser2_id": id_b,
                    "roaduser1_type": trk_a.cls_name, "roaduser2_type": trk_b.cls_name,
                    "encounter_type": encounter,
                    "crash_codes": crash_codes_for(encounter),
                    "governing_measure": governing,
                    "level": final,
                    "ttc_s": round(ttc, 3) if ttc is not None else None,
                    "pet_s": round(pet, 3) if pet is not None else None,
                    "pet_source": pet_src,
                    "drac_mps2": round(drac, 3) if drac is not None else None,
                    "deltav_mps": round(dv, 3) if dv is not None else None,
                    "metric_conditional": not metric,
                    "nfb3": int(nfb3), "nfb3_window": int(nwin),
                    "conflict_x_bev": round(float(cpt[0]), 3),
                    "conflict_y_bev": round(float(cpt[1]), 3),
                    "conflict_lat": round(conflict_geo["lat"], 8) if conflict_geo else None,
                    "conflict_lng": round(conflict_geo["lng"], 8) if conflict_geo else None,
                    "conflict_utm": conflict_geo["utm"] if conflict_geo else None,
                    "interaction_angle_deg": round(angle, 1),
                    "closing_speed_mps": round(closing, 2),
                    "relative_speed_mps": round(float(np.linalg.norm(va - vb)), 2),
                    "forecast_mode": f"{trk_a.forecast_mode()}/{trk_b.forecast_mode()}",
                    "zones": zones,
                    "vulnerable_involved": (trk_a.cls_name in cfg.vulnerable_classes or
                                            trk_b.cls_name in cfg.vulnerable_classes),
                    "confidence": round(float(min(trk_a.det_score, trk_b.det_score)), 3),
                })
                self.emit_cooldown[key] = self._dedupe

        self.nfb3.prune(live_pairs)
        self.frame_levels = frame_levels
        self.frame_active = frame_active
        self.frame_traces = frame_traces
        return incidents
