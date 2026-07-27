"""Tests for the RITSMS-aligned conflict engine (utils/near_miss.py):
vectorised TTC matrix, NFb3 confirmer, spatial-grid PET, and the integrated
evaluate() (TTC + PET + levels + dedupe).

Runnable with pytest or directly:  python tests/test_conflict_engine.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import AppConfig
from utils.motion import TrackState
from utils.near_miss import (
    NearMissEngine, AlertConfirmer, PETDetector, compute_ttc_matrix, LEVEL_ORDER,
)


# ── compute_ttc_matrix ───────────────────────────────────────────────────

def test_ttc_matrix_head_on_closing():
    pos = np.array([[0.0, 0.0], [20.0, 0.0]])
    vel = np.array([[5.0, 0.0], [-5.0, 0.0]])
    rad = np.array([1.0, 1.0])
    mx = compute_ttc_matrix(pos, vel, rad, 0.3, 3.0, 0.25, 0.1)
    assert math.isfinite(mx[0, 1]) and 1.5 < mx[0, 1] < 2.0
    assert mx[0, 1] == mx[1, 0]


def test_ttc_matrix_diverging_is_inf():
    pos = np.array([[0.0, 0.0], [20.0, 0.0]])
    vel = np.array([[-5.0, 0.0], [5.0, 0.0]])   # moving apart
    mx = compute_ttc_matrix(pos, vel, np.array([1.0, 1.0]), 0.3, 3.0, 0.25, 0.1)
    assert not math.isfinite(mx[0, 1])


def test_ttc_matrix_parallel_offset_is_inf():
    pos = np.array([[0.0, 0.0], [0.0, 50.0]])
    vel = np.array([[5.0, 0.0], [5.0, 0.0]])    # same velocity, far apart
    mx = compute_ttc_matrix(pos, vel, np.array([1.0, 1.0]), 0.3, 3.0, 0.25, 0.1)
    assert not math.isfinite(mx[0, 1])


# ── AlertConfirmer (NFb3 M-of-N) ─────────────────────────────────────────

def test_confirmer_requires_min_count():
    c = AlertConfirmer(window_size=3, min_count=2)
    assert c.update_frame({(1, 2)}) == set()          # 1 frame -> not yet
    assert (1, 2) in c.update_frame({(1, 2)})          # 2 frames -> confirmed


def test_confirmer_drops_stale():
    c = AlertConfirmer(window_size=2, min_count=2)
    c.update_frame({(1, 2)}); c.update_frame({(1, 2)})
    c.update_frame(set()); c.update_frame(set())       # window empties
    assert (1, 2) not in c._history


# ── PETDetector ──────────────────────────────────────────────────────────

def _stamp(tid, x, y, t, vx=5.0, vy=0.0, cls="car"):
    return {"id": tid, "t": t, "x": x, "y": y, "vx": vx, "vy": vy,
            "speed": math.hypot(vx, vy), "cls": cls, "rad": 0.9, "det": 0.9}


def test_pet_detector_arrival_gap():
    pet = PETDetector(cell_size_m=1.5, max_window_s=6.0)
    assert pet.visit_and_query(1, _stamp(1, 0.0, 0.0, 0.0)) == {}      # first visitor
    got = pet.visit_and_query(2, _stamp(2, 0.2, 0.1, 1.0))            # same cell, +1s
    assert set(got) == {1} and abs(got[1][0] - 1.0) < 1e-9


def test_pet_detector_evicts_beyond_window():
    pet = PETDetector(cell_size_m=1.5, max_window_s=2.0)
    pet.visit_and_query(1, _stamp(1, 0.0, 0.0, 0.0))
    got = pet.visit_and_query(2, _stamp(2, 0.0, 0.0, 5.0))            # 5s > 2s window
    assert got == {}


# ── integrated evaluate() ────────────────────────────────────────────────

def _mk(tid, pos, heading_deg, speed, frame_idx, cls="car", age=25):
    t = TrackState(track_id=tid, cls_name=cls)
    t.age = age
    t.det_score = 0.9
    t.last_bbox = [600, 300, 660, 360]
    t.last_update_frame = frame_idx
    t.bev_radius_m = 1.0
    t.kf_mean = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
    t.bev_positions.append(np.asarray(pos, dtype=np.float64))
    t.bev_positions.append(np.asarray(pos, dtype=np.float64))
    t.speed = float(speed)
    t.heading_deg = float(heading_deg)
    rad = np.radians(heading_deg)
    t.vel = np.array([speed * np.cos(rad), speed * np.sin(rad)], dtype=np.float64)
    return t


def test_evaluate_ttc_fires_after_confirmation():
    eng = NearMissEngine(AppConfig(), fps=10.0, site=None)  # window=5, min=4
    inc = []
    # head-on closing pair held over several frames -> NFb3 confirms.
    for f in range(1, 7):
        tracks = {1: _mk(1, (0, 0), 0, 5, f), 2: _mk(2, (20, 0), 180, 5, f)}
        inc = eng.evaluate(f, "t", tracks, f / 10.0)
    kinds = [e for e in inc if e.get("metric_kind", "").startswith("TTC")]
    # confirmed on an earlier frame then deduped; check across the run instead
    # by re-running and collecting all:
    eng2 = NearMissEngine(AppConfig(), fps=10.0, site=None)
    all_inc = []
    for f in range(1, 7):
        tracks = {1: _mk(1, (0, 0), 0, 5, f), 2: _mk(2, (20, 0), 180, 5, f)}
        all_inc += eng2.evaluate(f, "t", tracks, f / 10.0)
    ttc = [e for e in all_inc if "TTC" in e.get("metric_kind", "")]
    assert len(ttc) >= 1, all_inc
    assert ttc[0]["level"] in ("WARNING", "CRITICAL")
    assert ttc[0]["ttc_s"] is not None


def test_evaluate_ttc_dedupes():
    eng = NearMissEngine(AppConfig(), fps=10.0, site=None)
    all_inc = []
    for f in range(1, 20):
        tracks = {1: _mk(1, (0, 0), 0, 5, f), 2: _mk(2, (20, 0), 180, 5, f)}
        all_inc += eng.evaluate(f, "t", tracks, f / 10.0)
    ttc = [e for e in all_inc if "TTC" in e.get("metric_kind", "")]
    # dedupe_frames = 3s * 10fps = 30 > 19 frames -> at most one emit.
    assert len(ttc) == 1, f"expected 1 deduped TTC, got {len(ttc)}"


def test_evaluate_pet_crossing_fires():
    eng = NearMissEngine(AppConfig(), fps=10.0, site=None)
    # A crosses origin cell at frame 5 (+x); B arrives at frame 15 (+y).
    eng.evaluate(5, "t", {1: _mk(1, (0, 0), 0, 5, 5), 2: _mk(2, (0, -5), 90, 5, 5)}, 0.5)
    inc = eng.evaluate(15, "t", {1: _mk(1, (5, 0), 0, 5, 15), 2: _mk(2, (0, 0), 90, 5, 15)}, 1.5)
    pet = [e for e in inc if "PET" in e.get("metric_kind", "")]
    assert len(pet) == 1, inc
    assert pet[0]["pet_s"] == 1.0
    assert pet[0]["level"] == "CRITICAL"          # 1.0 < pet_critical 1.5
    assert pet[0]["scenario"] == "crossing"       # 90deg < opposing threshold


def test_evaluate_pet_departed_crosser():
    """The defining PET case: the earlier crosser has LEFT the scene by the
    time the second arrives. Must still fire, reconstructed from the stamp."""
    eng = NearMissEngine(AppConfig(), fps=10.0, site=None)
    eng.evaluate(5, "t", {1: _mk(1, (0, 0), 0, 5, 5), 2: _mk(2, (0, -5), 90, 5, 5)}, 0.5)
    # frame 15: track 1 is gone from `tracks`; only the arriver (2) remains.
    inc = eng.evaluate(15, "t", {2: _mk(2, (0, 0), 90, 5, 15)}, 1.5)
    pet = [e for e in inc if "PET" in e.get("metric_kind", "")]
    assert len(pet) == 1, inc
    assert {pet[0]["actor_1_id"], pet[0]["actor_2_id"]} == {1, 2}
    assert pet[0]["pet_s"] == 1.0


def test_evaluate_ignores_diverging_pair():
    eng = NearMissEngine(AppConfig(), fps=10.0, site=None)
    all_inc = []
    for f in range(1, 8):
        # moving apart, same direction, far -> no TTC, no crossing PET
        tracks = {1: _mk(1, (0, 0), 0, 5, f), 2: _mk(2, (30, 0), 0, 5, f)}
        all_inc += eng.evaluate(f, "t", tracks, f / 10.0)
    assert [e for e in all_inc if e.get("metric_kind", "") in ("TTC", "PET", "TTC+PET")] == []


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
