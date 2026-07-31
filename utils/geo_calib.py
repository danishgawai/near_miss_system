"""Metric (survey-grade) calibration from image <-> lat/lng correspondences.

This is the calibration path that lifts a site out of `bev_only` mode. Instead
of mapping pixels to an arbitrary BEV canvas and guessing a pixels-per-metre
scale, we map pixels directly to local East/North METRES on a tangent plane
built from real-world lat/lng ground-control points:

    1. lat/lng GCPs -> local metres on an equirectangular plane centred on the
       mean of the world points (lat0, lng0).
    2. cv2.findHomography(image_px -> metres) gives H; H^-1 maps back.
    3. Reprojection error is therefore measured IN METRES, which is exactly what
       proposal AC1 asks for (RMSE + P95).
    4. Positions can be inverted to lat/lng and UTM for georeferenced outputs.

Because the plane is metric, every threshold (TTC/PET seconds, DRAC m/s^2,
Delta-V m/s) is physically meaningful and transfers across cameras unchanged.

Calibration JSON format (as produced by the IP Camera Locator tool):

    {
      "camera": {"lat": 24.445636, "lng": 54.401949, "ip": "IP86", ...},
      "video":  {"width": 1280, "height": 720, "filename": "..."},
      "correspondences": [
        {"id": 1, "image": {"x": 155, "y": 345},
                  "world": {"lat": 24.4456338, "lng": 54.4016240}},
        ...
      ]
    }
"""

import json
import math
import logging
import numpy as np
import cv2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

EARTH_R_M = 6378137.0

# WGS84 / UTM constants for the optional UTM block in outputs.
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F * _WGS84_F
_WGS84_EP2 = _WGS84_E2 / (1.0 - _WGS84_E2)
_UTM_K0 = 0.9996


def latlng_to_utm(lat_deg: float, lng_deg: float) -> Optional[dict]:
    """WGS84 lat/lng -> UTM easting/northing (transverse Mercator series)."""
    if lat_deg is None or lng_deg is None or not (-80.0 <= lat_deg <= 84.0):
        return None
    zone = max(1, min(60, int((lng_deg + 180.0) / 6.0) + 1))
    lon0_deg = (zone - 1) * 6.0 - 180.0 + 3.0

    lat, lng, lon0 = math.radians(lat_deg), math.radians(lng_deg), math.radians(lon0_deg)
    sin_lat, cos_lat, tan_lat = math.sin(lat), math.cos(lat), math.tan(lat)
    e2, ep2 = _WGS84_E2, _WGS84_EP2

    N = _WGS84_A / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    T = tan_lat * tan_lat
    C = ep2 * cos_lat * cos_lat
    A = (lng - lon0) * cos_lat
    M = _WGS84_A * (
        (1.0 - e2 / 4.0 - 3.0 * e2 ** 2 / 64.0 - 5.0 * e2 ** 3 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e2 ** 2 / 32.0 + 45.0 * e2 ** 3 / 1024.0) * math.sin(2.0 * lat)
        + (15.0 * e2 ** 2 / 256.0 + 45.0 * e2 ** 3 / 1024.0) * math.sin(4.0 * lat)
        - (35.0 * e2 ** 3 / 3072.0) * math.sin(6.0 * lat)
    )
    A2, A3 = A * A, A ** 3
    A4, A5, A6 = A ** 4, A ** 5, A ** 6
    easting = _UTM_K0 * N * (
        A + (1.0 - T + C) * A3 / 6.0
        + (5.0 - 18.0 * T + T * T + 72.0 * C - 58.0 * ep2) * A5 / 120.0
    ) + 500000.0
    northing = _UTM_K0 * (
        M + N * tan_lat * (
            A2 / 2.0 + (5.0 - T + 9.0 * C + 4.0 * C * C) * A4 / 24.0
            + (61.0 - 58.0 * T + T * T + 600.0 * C - 330.0 * ep2) * A6 / 720.0
        )
    )
    if lat_deg < 0.0:
        northing += 10000000.0
    return {"easting": float(easting), "northing": float(northing),
            "zone": int(zone), "hemisphere": "N" if lat_deg >= 0.0 else "S"}


class GeoCalibration:
    """Homography from image pixels to local East/North metres, with geo inverse.

    Attributes set on success:
        H          3x3 image px -> metres
        H_inv      3x3 metres -> image px
        rmse_m     reprojection RMSE over the GCPs, in metres (AC1)
        p95_m      95th-percentile reprojection error, metres (AC1)
        max_m      worst GCP residual, metres
    """

    def __init__(self, src_px, dst_latlng, camera=None, video=None):
        src = np.asarray(src_px, dtype=np.float64)
        dst = np.asarray(dst_latlng, dtype=np.float64)
        if len(src) < 4 or len(src) != len(dst):
            raise ValueError(f"need >= 4 matched correspondences, got {len(src)}/{len(dst)}")
        if np.abs(dst[:, 0]).max() > 90.0 or np.abs(dst[:, 1]).max() > 180.0:
            raise ValueError("world values are not valid lat/lng degrees")

        self.camera = camera or {}
        self.video = video or {}
        self.src_px = src

        # Tangent plane centred on the mean of the GCPs.
        self.lat0 = float(dst[:, 0].mean())
        self.lng0 = float(dst[:, 1].mean())
        self.cos_lat0 = math.cos(math.radians(self.lat0))

        dst_m = np.array([self.latlng_to_local_m(lat, lng) for lat, lng in dst],
                         dtype=np.float64)
        self.dst_m = dst_m

        H, _ = cv2.findHomography(src.astype(np.float32), dst_m.astype(np.float32), method=0)
        if H is None:
            raise RuntimeError("cv2.findHomography failed on the supplied correspondences")
        self.H = np.asarray(H, dtype=np.float64)
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            raise RuntimeError("homography is singular; check the correspondences")

        # Reprojection error IN METRES -- proposal AC1 evidence.
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32),
                                        self.H.astype(np.float32)).reshape(-1, 2)
        resid = np.linalg.norm(proj - dst_m, axis=1)
        self.residuals_m = resid
        self.rmse_m = float(np.sqrt(np.mean(resid ** 2)))
        self.p95_m = float(np.percentile(resid, 95))
        self.max_m = float(resid.max())

        # Sign of the projective denominator over the calibrated region; points
        # beyond the horizon flip sign and must be rejected.
        c = src.mean(axis=0)
        w = self.H[2, 0] * c[0] + self.H[2, 1] * c[1] + self.H[2, 2]
        self._w_sign = 1.0 if w >= 0 else -1.0

        logger.info("Metric calibration: %d GCPs, ref=(%.6f, %.6f), "
                    "reprojection RMSE=%.3f m P95=%.3f m max=%.3f m",
                    len(src), self.lat0, self.lng0, self.rmse_m, self.p95_m, self.max_m)

    # -- geo conversions --
    def latlng_to_local_m(self, lat: float, lng: float) -> Tuple[float, float]:
        east = EARTH_R_M * math.radians(lng - self.lng0) * self.cos_lat0
        north = EARTH_R_M * math.radians(lat - self.lat0)
        return east, north

    def local_m_to_latlng(self, east_m: float, north_m: float) -> Tuple[float, float]:
        lat = self.lat0 + math.degrees(north_m / EARTH_R_M)
        lng = self.lng0 + math.degrees(east_m / (EARTH_R_M * self.cos_lat0))
        return lat, lng

    # -- projections --
    def to_metres(self, x: float, y: float) -> Optional[np.ndarray]:
        """Image pixel -> local East/North metres. None beyond the horizon."""
        w = self.H[2, 0] * x + self.H[2, 1] * y + self.H[2, 2]
        if w * self._w_sign < 1e-9:
            return None
        east = (self.H[0, 0] * x + self.H[0, 1] * y + self.H[0, 2]) / w
        north = (self.H[1, 0] * x + self.H[1, 1] * y + self.H[1, 2]) / w
        out = np.array([east, north], dtype=np.float64)
        return out if np.all(np.isfinite(out)) else None

    def to_image_px(self, east_m: float, north_m: float) -> Optional[Tuple[float, float]]:
        v = self.H_inv @ np.array([east_m, north_m, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-12:
            return None
        px, py = v[0] / v[2], v[1] / v[2]
        return (float(px), float(py)) if np.isfinite(px) and np.isfinite(py) else None

    def geo_for(self, east_m: float, north_m: float) -> dict:
        """lat/lng (+UTM) for a metric position -- georeferenced outputs."""
        lat, lng = self.local_m_to_latlng(east_m, north_m)
        return {"lat": lat, "lng": lng, "utm": latlng_to_utm(lat, lng)}

    def summary(self) -> dict:
        return {
            "n_gcps": int(len(self.src_px)),
            "ref_lat": self.lat0, "ref_lng": self.lng0,
            "reprojection_rmse_m": round(self.rmse_m, 4),
            "reprojection_p95_m": round(self.p95_m, 4),
            "reprojection_max_m": round(self.max_m, 4),
            "camera_ip": self.camera.get("ip"),
            "camera_label": self.camera.get("label"),
        }


def load_gcp_calibration(path: str) -> Optional[GeoCalibration]:
    """Build a GeoCalibration from a correspondences JSON. None if the file has
    no lat/lng correspondences (i.e. it is a plain BEV-canvas config)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Cannot read calibration '%s': %s", path, e)
        return None

    corrs = data.get("correspondences")
    if not corrs:
        return None                      # not a GCP calibration
    src, dst = [], []
    for c in corrs:
        try:
            src.append([float(c["image"]["x"]), float(c["image"]["y"])])
            dst.append([float(c["world"]["lat"]), float(c["world"]["lng"])])
        except (KeyError, TypeError, ValueError):
            continue
    if len(src) < 4:
        logger.warning("Calibration '%s' has %d usable correspondences (need >= 4)",
                       path, len(src))
        return None
    try:
        return GeoCalibration(src, dst, data.get("camera"), data.get("video"))
    except Exception as e:
        logger.error("Metric calibration failed for '%s': %s", path, e)
        return None
