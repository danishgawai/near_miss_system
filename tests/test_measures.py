"""Unit tests for utils/measures.py — the proposal §4.5 conflict measures.

Runnable two ways:
    pytest tests/test_measures.py
    python tests/test_measures.py        (no pytest needed)
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils import measures as m


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ── 1-D TTC ─────────────────────────────────────────────────────────────────

def test_ttc_rear_end_closing():
    # leader at 20, follower at 0, follower faster by 5 m/s, 4 m car length.
    # gap = 20 - 0 - 4 = 16, closing = 5 -> 3.2 s
    assert _close(m.ttc_rear_end(20, 0, 5, 10, d_lead=4), 3.2)


def test_ttc_rear_end_not_closing():
    # follower slower than leader -> no rear-end TTC
    assert m.ttc_rear_end(20, 0, 10, 5, d_lead=4) is None


def test_ttc_rear_end_gap_collapsed():
    assert _close(m.ttc_rear_end(2, 0, 5, 10, d_lead=4), 0.0)


def test_ttc_head_on():
    # 30 m apart, closing at 5 + 5 = 10 m/s -> 3.0 s
    assert _close(m.ttc_head_on(30, 0, 5, 5), 3.0)


def test_ttc_head_on_not_approaching():
    assert m.ttc_head_on(30, 0, 0, 0) is None


# ── interaction angle ─────────────────────────────────────────────────────────

def test_interaction_angle_head_on():
    assert _close(m.interaction_angle([1, 0], [-1, 0]), math.pi, tol=1e-9)


def test_interaction_angle_same_dir():
    assert _close(m.interaction_angle([2, 0], [5, 0]), 0.0, tol=1e-9)


def test_interaction_angle_right_angle():
    assert _close(m.interaction_angle([1, 0], [0, 1]), math.pi / 2, tol=1e-9)


def test_interaction_angle_stationary():
    assert _close(m.interaction_angle([0, 0], [1, 1]), 0.0)


# ── 2-D AABB TTC ──────────────────────────────────────────────────────────────

def test_ttc_2d_aabb_head_on():
    # unit-half squares; R1 fixed at origin, R2 at (10,0) moving left at 5 m/s.
    # near edges 8 m apart, closing 5 -> 1.6 s
    ttc = m.ttc_2d_aabb([0, 0], [0, 0], [1, 1], [10, 0], [-5, 0], [1, 1])
    assert ttc is not None and _close(ttc, 1.6)


def test_ttc_2d_aabb_parallel_no_hit():
    # moving along +x but offset far in y, never overlapping in y
    ttc = m.ttc_2d_aabb([0, 0], [0, 0], [1, 1], [10, 50], [-5, 0], [1, 1])
    assert ttc is None


def test_ttc_2d_aabb_already_overlapping():
    ttc = m.ttc_2d_aabb([0, 0], [0, 0], [2, 2], [1, 0], [-1, 0], [2, 2])
    assert ttc is not None and _close(ttc, 0.0)


def test_ttc_2d_aabb_diverging():
    # R2 to the right moving further right -> never hits
    ttc = m.ttc_2d_aabb([0, 0], [0, 0], [1, 1], [10, 0], [5, 0], [1, 1])
    assert ttc is None


# ── 2-D OBB TTC ───────────────────────────────────────────────────────────────

def test_ttc_2d_obb_matches_aabb_at_zero_heading():
    # axis-aligned oriented rects should agree with the AABB result (~within dt)
    aabb = m.ttc_2d_aabb([0, 0], [0, 0], [1, 1], [10, 0], [-5, 0], [1, 1])
    obb = m.ttc_2d_obb([0, 0], [0, 0], 0.0, 2, 2, [10, 0], [-5, 0], 0.0, 2, 2, dt=0.02)
    assert obb is not None and abs(obb - aabb) <= 0.05


def test_ttc_2d_obb_no_contact():
    obb = m.ttc_2d_obb([0, 0], [0, 0], 0.0, 2, 2, [10, 50], [-5, 0], 0.0, 2, 2)
    assert obb is None


# ── PET ───────────────────────────────────────────────────────────────────────

def test_pet_positive_gap():
    assert _close(m.pet(2.0, 3.5), 1.5)


def test_pet_temporal_overlap_negative():
    # second arrives before first leaves -> non-positive -> collision course
    assert m.pet(3.0, 2.0) < 0


# ── DRAC ──────────────────────────────────────────────────────────────────────

def test_drac_value():
    # dv = 5, gap = 20 - 0 - 4 = 16 -> 25 / 32 = 0.78125
    assert _close(m.drac(20, 0, 5, 10, d_lead=4), 25.0 / 32.0)


def test_drac_not_closing():
    assert m.drac(20, 0, 10, 5, d_lead=4) is None


# ── Delta-V ────────────────────────────────────────────────────────────────────

def test_delta_v_head_on_equal_mass():
    # equal-mass head-on at 5 m/s each: perfectly inelastic -> both stop,
    # ΔV = 5 m/s for each; rel speed = 10, ΔV = 1000*10/2000 = 5
    assert _close(m.delta_v(5, 5, 1000, 1000, math.pi), 5.0)


def test_delta_v_mass_ratio():
    # heavier vehicle experiences the smaller ΔV; reported value is the max
    # (the lighter vehicle's). ΔV_light / ΔV_heavy = m_heavy / m_light.
    dv = m.delta_v(6, 6, 1000, 3000, math.pi)  # rel = 12
    # light (m_f=1000) ΔV uses m_s=3000: 3000*12/4000 = 9.0
    assert _close(dv, 9.0)


def test_delta_v_both_stationary():
    assert _close(m.delta_v(0, 0, 1000, 2000, 0.0), 0.0)


# ── runner (no pytest) ──────────────────────────────────────────────────────

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
