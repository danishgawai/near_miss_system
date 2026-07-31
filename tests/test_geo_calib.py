"""Tests for utils/geo_calib.py and the crash-code taxonomy.

    python tests/test_geo_calib.py
"""

import os
import sys
import json
import math
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.geo_calib import (
    GeoCalibration, load_gcp_calibration, latlng_to_utm, EARTH_R_M,
)
from ritsms.patterns import crash_codes_for, CRASH_CODES, NOT_DETECTED


# A synthetic but geometrically exact calibration: four image corners mapped to
# four lat/lng points forming a known rectangle on the ground near Abu Dhabi.
LAT0, LNG0 = 24.445636, 54.401949


def _latlng_offset(lat0, lng0, east_m, north_m):
    lat = lat0 + math.degrees(north_m / EARTH_R_M)
    lng = lng0 + math.degrees(east_m / (EARTH_R_M * math.cos(math.radians(lat0))))
    return lat, lng


def _exact_calib():
    # 40 m x 20 m ground rectangle, mapped from a 4-point image quad.
    src = [[100.0, 600.0], [500.0, 600.0], [500.0, 300.0], [100.0, 300.0]]
    corners_m = [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)]
    dst = [list(_latlng_offset(LAT0, LNG0, e, n)) for e, n in corners_m]
    return GeoCalibration(src, dst, camera={"ip": "TEST"}, video={"width": 1280, "height": 720})


# ── construction / AC1 error reporting ────────────────────────────────────

def test_builds_and_reports_low_reprojection_error():
    g = _exact_calib()
    # An exact 4-point mapping must reproject to sub-millimetre residuals
    # (not exactly zero: float32 homography + equirectangular approximation).
    assert g.rmse_m < 1e-3, g.rmse_m
    assert g.p95_m < 1e-3
    s = g.summary()
    assert s["n_gcps"] == 4 and "reprojection_rmse_m" in s and "reprojection_p95_m" in s


def test_rejects_too_few_correspondences():
    try:
        GeoCalibration([[0, 0], [1, 1], [2, 2]], [[LAT0, LNG0]] * 3)
    except ValueError as e:
        assert "correspondences" in str(e)
    else:
        raise AssertionError("should reject < 4 points")


def test_rejects_non_latlng_world_values():
    src = [[0, 0], [1, 0], [1, 1], [0, 1]]
    dst = [[1000.0, 2000.0]] * 4          # clearly not degrees
    try:
        GeoCalibration(src, dst)
    except ValueError as e:
        assert "lat/lng" in str(e)
    else:
        raise AssertionError("should reject non-degree world values")


# ── projection correctness ────────────────────────────────────────────────

def test_pixel_to_metres_matches_known_rectangle():
    g = _exact_calib()
    # The tangent plane is centred on the CENTROID of the GCPs, so a 40 x 20 m
    # ground rectangle spans (-20,-10) .. (+20,+10) rather than starting at 0.
    for px, expect in zip([[100, 600], [500, 600], [500, 300], [100, 300]],
                          [(-20, -10), (20, -10), (20, 10), (-20, 10)]):
        m = g.to_metres(px[0], px[1])
        assert m is not None
        assert abs(m[0] - expect[0]) < 1e-3 and abs(m[1] - expect[1]) < 1e-3, (px, m)


def test_side_lengths_are_metrically_correct():
    # The point of metric calibration: measured distances are true metres.
    g = _exact_calib()
    bl = g.to_metres(100, 600); br = g.to_metres(500, 600); tl = g.to_metres(100, 300)
    assert abs(float(np.linalg.norm(br - bl)) - 40.0) < 1e-3
    assert abs(float(np.linalg.norm(tl - bl)) - 20.0) < 1e-3


def test_metres_to_pixel_round_trip():
    g = _exact_calib()
    m = g.to_metres(300.0, 450.0)
    back = g.to_image_px(m[0], m[1])
    assert abs(back[0] - 300.0) < 1e-6 and abs(back[1] - 450.0) < 1e-6


def test_latlng_round_trip():
    g = _exact_calib()
    e, n = 12.5, -7.25
    lat, lng = g.local_m_to_latlng(e, n)
    e2, n2 = g.latlng_to_local_m(lat, lng)
    assert abs(e2 - e) < 1e-6 and abs(n2 - n) < 1e-6


def test_geo_for_returns_latlng_and_utm():
    g = _exact_calib()
    out = g.geo_for(10.0, 5.0)
    assert "lat" in out and "lng" in out and out["utm"] is not None
    assert abs(out["lat"] - LAT0) < 0.01 and abs(out["lng"] - LNG0) < 0.01


# ── UTM ────────────────────────────────────────────────────────────────────

def test_utm_zone_and_hemisphere():
    u = latlng_to_utm(24.445636, 54.401949)          # Abu Dhabi -> zone 40 N
    assert u["zone"] == 40 and u["hemisphere"] == "N"
    assert 100000 < u["easting"] < 900000 and u["northing"] > 0


def test_utm_southern_hemisphere_offset():
    u = latlng_to_utm(-33.86, 151.21)                 # Sydney
    assert u["hemisphere"] == "S" and u["northing"] > 1_000_000


def test_utm_rejects_extreme_latitude():
    assert latlng_to_utm(89.0, 0.0) is None


# ── loader ────────────────────────────────────────────────────────────────

def _write(data):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def test_loader_reads_gcp_file():
    src = [[100, 600], [500, 600], [500, 300], [100, 300]]
    corners = [(0, 0), (40, 0), (40, 20), (0, 20)]
    corrs = []
    for i, (px, (e, n)) in enumerate(zip(src, corners), 1):
        lat, lng = _latlng_offset(LAT0, LNG0, e, n)
        corrs.append({"id": i, "image": {"x": px[0], "y": px[1]},
                      "world": {"lat": lat, "lng": lng}})
    p = _write({"camera": {"ip": "IP86"}, "video": {"width": 1280, "height": 720},
                "correspondences": corrs})
    try:
        g = load_gcp_calibration(p)
        assert g is not None and g.rmse_m < 1e-3
        assert g.camera.get("ip") == "IP86"
    finally:
        os.remove(p)


def test_loader_returns_none_for_plain_bev_config():
    # A canvas-style bev_config.json has no `correspondences` -> not metric.
    p = _write({"homography_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "pixels_per_meter": 32.8})
    try:
        assert load_gcp_calibration(p) is None
    finally:
        os.remove(p)


# ── crash codes ───────────────────────────────────────────────────────────

def test_crash_codes_for_known_encounters():
    assert crash_codes_for("rear_end") == ["4.1", "4.2", "4.3"]
    assert crash_codes_for("lane_change_merge") == ["4.4", "4.5", "4.6"]
    assert crash_codes_for("head_on") == ["3.1"]
    assert crash_codes_for("opposing_through") == ["3.2", "3.3"]


def test_crash_codes_unknown_encounter_is_empty():
    assert crash_codes_for("something_else") == []


def test_crash_codes_returns_a_copy():
    got = crash_codes_for("rear_end")
    got.append("9.9")
    assert crash_codes_for("rear_end") == ["4.1", "4.2", "4.3"]


def test_undetected_families_are_documented():
    # Scoped out explicitly rather than silently missing.
    assert NOT_DETECTED["u_turn"] == ["5.6"]
    assert NOT_DETECTED["parallel_lane_turning"] == ["4.7", "4.8"]
    for fam in NOT_DETECTED:
        assert fam not in CRASH_CODES


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
