"""Tests for ritsms/calibration_check.py — the two calibration guards.

    python tests/test_calibration_check.py
"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ritsms.calibration_check import check_config_match, CalibrationHealth


def _cfg_file(**over):
    base = {
        "video_source": "site_A.mp4",
        "frame_width": 1280, "frame_height": 720,
        "homography_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    base.update(over)
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(base, f)
    return path


# ── static config match ───────────────────────────────────────────────────

def test_match_ok():
    p = _cfg_file()
    try:
        assert check_config_match(p, "site_A.mp4", 1280, 720) == []
    finally:
        os.remove(p)


def test_match_accepts_different_directory():
    p = _cfg_file()
    try:
        # Paths legitimately differ between calibration and run; basename matters.
        assert check_config_match(p, "/data/videos/site_A.mp4", 1280, 720) == []
    finally:
        os.remove(p)


def test_match_detects_wrong_video():
    p = _cfg_file()
    try:
        probs = check_config_match(p, "site_B.mp4", 1280, 720)
        assert len(probs) == 1 and "different camera view" in probs[0]
    finally:
        os.remove(p)


def test_match_detects_resolution_change():
    p = _cfg_file()
    try:
        probs = check_config_match(p, "site_A.mp4", 1920, 1080)
        assert any("1920x1080" in x for x in probs)
    finally:
        os.remove(p)


def test_match_detects_missing_homography():
    p = _cfg_file(homography_matrix=None)
    try:
        assert any("homography" in x for x in check_config_match(p, "site_A.mp4", 1280, 720))
    finally:
        os.remove(p)


def test_match_detects_missing_file():
    assert check_config_match("does_not_exist.json", "x.mp4", 1280, 720)


# ── empirical health ──────────────────────────────────────────────────────

class _Trk:
    """Minimal stand-in for TrackTrajectory."""
    def __init__(self, xy, speed):
        self._p = np.asarray(xy, dtype=np.float64)
        self.speed = speed
        self.ready = True

    @property
    def position_m(self):
        return self._p


def _feed(health, series):
    """series: {tid: [(x, y, speed), ...]} fed frame by frame."""
    n = max(len(v) for v in series.values())
    for i in range(n):
        frame = {}
        for tid, pts in series.items():
            if i < len(pts):
                x, y, s = pts[i]
                frame[tid] = _Trk((x, y), s)
        health.observe(frame)


def test_health_pass_when_stopped_tracks_are_still():
    h = CalibrationHealth()
    # three stopped vehicles barely move; a mover spans the scene
    series = {i: [(10.0 + i * 3 + 0.02 * k, 5.0, 0.05) for k in range(30)] for i in (1, 2, 3)}
    series[9] = [(0.0 + 1.5 * k, 20.0, 8.0) for k in range(30)]
    _feed(h, series)
    rep = h.report()
    wander = [c for c in rep["checks"] if c["check"] == "stopped_track_bev_wander"][0]
    assert wander["status"] == "INFO", rep
    assert rep["overall"] in ("PASS", "WARN"), rep


def test_wander_is_informational_only():
    # Measured against real known-good/known-bad calibrations this metric did NOT
    # separate them, so it must never drive a verdict - only report a number.
    h = CalibrationHealth()
    series = {i: [(10.0 + i * 3 + 0.9 * k, 5.0 + 0.5 * k, 0.05) for k in range(30)]
              for i in (1, 2, 3)}
    series[9] = [(0.0 + 1.5 * k, 20.0, 8.0) for k in range(30)]
    _feed(h, series)
    rep = h.report()
    wander = [c for c in rep["checks"] if c["check"] == "stopped_track_bev_wander"][0]
    assert wander["status"] == "INFO", rep
    assert wander["median_m"] > 4.0                 # large drift still reported
    assert rep["overall"] != "FAIL"                 # ...but does not fail the run


def test_health_ignores_approach_distance_of_vehicle_that_stops():
    # A vehicle decelerating into a queue legitimately covers its approach
    # distance. Only the frames where it is actually stopped may be measured,
    # otherwise a correct calibration is wrongly failed.
    h = CalibrationHealth()
    series = {}
    for i in (1, 2, 3):
        approach = [(i * 3 + 1.5 * k, 5.0, 8.0) for k in range(12)]        # moving
        halted = [(i * 3 + 1.5 * 11 + 0.01 * k, 5.0, 0.05) for k in range(15)]  # stopped
        series[i] = approach + halted
    _feed(h, series)
    rep = h.report()
    wander = [c for c in rep["checks"] if c["check"] == "stopped_track_bev_wander"][0]
    assert wander["status"] == "INFO", rep


def test_health_fails_on_compressed_scene_span():
    h = CalibrationHealth(min_scene_span_m=20.0)
    tiny = [(1.0 + 0.05 * i, 1.0, 0.05) for i in range(30)]     # ~1.5 m of scene
    _feed(h, {1: tiny})
    rep = h.report()
    span = [c for c in rep["checks"] if c["check"] == "scene_span"][0]
    assert span["status"] == "FAIL", rep


def test_health_flags_implausible_speed():
    h = CalibrationHealth(max_speed_mps=40.0)
    fast = [(0.0 + 10.0 * i, 0.0, 95.0) for i in range(30)]
    _feed(h, {1: fast})
    rep = h.report()
    sp = [c for c in rep["checks"] if c["check"] == "speed_plausibility"][0]
    assert sp["status"] == "FAIL", rep


def test_health_skips_wander_with_no_stopped_tracks():
    h = CalibrationHealth()
    mover = [(0.0 + 1.5 * i, 20.0, 8.0) for i in range(30)]
    _feed(h, {1: mover})
    rep = h.report()
    wander = [c for c in rep["checks"] if c["check"] == "stopped_track_bev_wander"][0]
    assert wander["status"] == "SKIP", rep


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
