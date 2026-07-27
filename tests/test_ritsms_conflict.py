"""Tests for ritsms/conflict.py — forecast-driven TTC/PET/DRAC/ΔV engine.

    python tests/test_ritsms_conflict.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ritsms.config import Config
from ritsms.trajectory import TrackTrajectory
from ritsms.conflict import (
    ConflictEngine, predictive_ttc, predictive_pet, PETGrid,
)


def _ftrack(tid, pos, vel, cfg, cls="car", yaw_dps=0.0):
    t = TrackTrajectory(track_id=tid, cls_name=cls, cfg=cfg)
    t._mean = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float64)
    t._cov = np.eye(4)
    t.n_updates = 5
    t.bev_positions.append(np.array([pos[0], pos[1]]))
    t.bev_positions.append(np.array([pos[0] + vel[0] * 0.1, pos[1] + vel[1] * 0.1]))
    t.speed = math.hypot(*vel)
    t.heading_deg = math.degrees(math.atan2(vel[1], vel[0]))
    t.yaw_rate = math.radians(yaw_dps)
    t.direction_consistency = 0.9
    t.age = 20
    t.det_score = 0.9
    t.last_bbox = [600, 300, 660, 360]
    return t


def _drive(eng, tracks, frames, fps=10.0):
    out = []
    for f in range(1, frames + 1):
        for t in tracks.values():
            t.last_update_frame = f
        out += eng.evaluate(f, "t", None, tracks, f / fps)
    return out


# ── low-level geometry ────────────────────────────────────────────────────

def test_predictive_ttc_head_on():
    cfg = Config()
    a = _ftrack(1, (0, 0), (5, 0), cfg)
    b = _ftrack(2, (20, 0), (-5, 0), cfg)
    ttc = predictive_ttc(a.forecast(), b.forecast(), a.length_m, a.width_m,
                         b.length_m, b.width_m, cfg.footprint_buffer_m)
    assert ttc is not None and 1.0 < ttc < 2.0, ttc


def test_predictive_pet_crossing_gap():
    cfg = Config()
    a = _ftrack(1, (0, 0), (5, 0), cfg)          # reaches (10,0) at t=2
    b = _ftrack(2, (10, -2.5), (0, 5), cfg)      # reaches (10,0) at t=0.5
    pp = predictive_pet(a.forecast(), b.forecast())
    assert pp is not None and abs(pp[0] - 1.5) < 0.2, pp


def test_predictive_pet_no_crossing_when_parallel():
    cfg = Config()
    a = _ftrack(1, (0, 0), (5, 0), cfg)          # parallel, offset in y
    b = _ftrack(2, (0, 8), (5, 0), cfg)
    assert predictive_pet(a.forecast(), b.forecast()) is None


def test_pet_grid_departed_crosser():
    g = PETGrid(1.5, 6.0)
    g.visit({"id": 1, "t": 0.0, "x": 0, "y": 0, "cls": "car"})
    got = g.visit({"id": 2, "t": 1.0, "x": 0.2, "y": 0.1, "cls": "car"})
    assert 1 in got and abs(got[1][0] - 1.0) < 1e-9


# ── integrated engine ──────────────────────────────────────────────────────

def test_engine_ttc_headon_confirms():
    cfg = Config()
    eng = ConflictEngine(cfg, fps=10.0, site=None)  # NFb3 window=5, min=4
    tracks = {1: _ftrack(1, (0, 0), (5, 0), cfg), 2: _ftrack(2, (20, 0), (-5, 0), cfg)}
    inc = _drive(eng, tracks, frames=8)
    assert len(inc) == 1, f"expected 1 deduped TTC conflict, got {len(inc)}"
    e = inc[0]
    assert e["ttc_s"] is not None and e["level"] in ("WARNING", "CRITICAL")
    assert e["nfb3"] >= 4


def test_engine_pet_crossing_fires():
    cfg = Config()
    eng = ConflictEngine(cfg, fps=10.0, site=None)
    tracks = {1: _ftrack(1, (0, 0), (5, 0), cfg), 2: _ftrack(2, (10, -2.5), (0, 5), cfg)}
    inc = _drive(eng, tracks, frames=2)
    pet = [e for e in inc if e["pet_s"] is not None]
    assert len(pet) >= 1, inc
    assert pet[0]["level"] in ("WARNING", "CRITICAL")
    assert "crossing" in pet[0]["encounter_type"]


def test_frame_levels_persist_while_unsafe():
    # Box highlighting (frame_levels) must persist EVERY frame the pair is
    # unsafe, even though the emitted record is deduped to one row.
    cfg = Config()
    eng = ConflictEngine(cfg, fps=10.0, site=None)
    tracks = {1: _ftrack(1, (0, 0), (5, 0), cfg), 2: _ftrack(2, (20, 0), (-5, 0), cfg)}
    highlighted, total = 0, 0
    for f in range(1, 12):
        for t in tracks.values():
            t.last_update_frame = f
        total += len(eng.evaluate(f, "t", None, tracks, f / 10.0))
        if eng.frame_levels:
            highlighted += 1
    assert total == 1, f"records should be deduped to 1, got {total}"
    assert highlighted >= 6, f"highlight should persist many frames, got {highlighted}"


def test_engine_ignores_slow_queue():
    cfg = Config()
    eng = ConflictEngine(cfg, fps=10.0, site=None)
    # same direction, follower barely faster (<2 m/s closing), 6 m apart -> no TTC
    tracks = {1: _ftrack(1, (0, 0), (2.0, 0), cfg), 2: _ftrack(2, (6, 0), (1.2, 0), cfg)}
    inc = _drive(eng, tracks, frames=8)
    assert inc == [], inc


def test_engine_ignores_far_apart():
    cfg = Config()
    eng = ConflictEngine(cfg, fps=10.0, site=None)
    tracks = {1: _ftrack(1, (0, 0), (5, 0), cfg), 2: _ftrack(2, (100, 100), (-5, 0), cfg)}
    inc = _drive(eng, tracks, frames=6)
    assert inc == []


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{p}/{len(fns)} passed")
    return p == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
