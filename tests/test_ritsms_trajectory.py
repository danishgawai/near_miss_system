"""Tests for ritsms/trajectory.py — the CV/CTRV forecast core.

    python tests/test_ritsms_trajectory.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ritsms.config import Config
from ritsms.trajectory import TrackTrajectory


def _make(vx, vy, yaw_dps, cls="car"):
    """A trajectory forced into a known kinematic state."""
    c = Config()
    t = TrackTrajectory(track_id=1, cls_name=cls, cfg=c)
    t._mean = np.array([0.0, 0.0, vx, vy], dtype=np.float64)
    t._cov = np.eye(4)
    t.n_updates = 5
    t.bev_positions.append(np.array([0.0, 0.0]))
    t.bev_positions.append(np.array([vx * 0.1, vy * 0.1]))
    t.speed = math.hypot(vx, vy)
    t.heading_deg = math.degrees(math.atan2(vy, vx))
    t.yaw_rate = math.radians(yaw_dps)
    t.direction_consistency = 0.9
    return t


def test_forecast_cv_straight():
    t = _make(5.0, 0.0, yaw_dps=0.0)              # straight, no yaw
    assert t.forecast_mode() == "CV"
    fc = t.forecast(horizon_s=2.0, dt_s=0.5)
    # rows: t, x, y, theta ; at t=1s x should be 5, y ~0, theta ~0
    row1 = fc[fc[:, 0] == 1.0][0]
    assert abs(row1[1] - 5.0) < 1e-6 and abs(row1[2]) < 1e-9
    assert abs(row1[3]) < 1e-9


def test_forecast_ctrv_curves():
    t = _make(5.0, 0.0, yaw_dps=30.0)             # turning left at 30 deg/s
    assert t.forecast_mode() == "CTRV"
    fc = t.forecast(horizon_s=2.0, dt_s=0.5)
    # heading must increase (turning), path must bend off the x-axis (+y)
    assert fc[-1, 3] > fc[0, 3]
    assert fc[-1, 2] > 0.2
    # forward progress still positive
    assert fc[-1, 1] > 0.0


def test_ctrv_falls_back_to_cv_when_slow():
    t = _make(0.3, 0.0, yaw_dps=30.0)             # below ctrv_min_speed
    assert t.forecast_mode() == "CV"


def test_ctrv_falls_back_to_cv_low_yaw():
    t = _make(5.0, 0.0, yaw_dps=2.0)              # below ctrv_min_yaw_rate
    assert t.forecast_mode() == "CV"


def test_forecast_length_and_horizon():
    t = _make(5.0, 0.0, yaw_dps=0.0)
    fc = t.forecast(horizon_s=3.0, dt_s=0.1)
    assert fc[0, 0] == 0.0 and abs(fc[-1, 0] - 3.0) < 1e-6
    assert fc.shape[1] == 4


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
