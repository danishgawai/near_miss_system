"""Bird's-Eye-View projection: image pixels -> metric ground-plane coordinates.

The homography H maps image pixels to BEV-canvas pixels; pixels_per_meter is
the scale OF THE BEV CANVAS (not the source image, where scale varies with
depth). Metric coordinates are therefore (H @ p) / ppm.

Validity handling: a ground homography is only defined below the camera's
horizon line. As a pixel approaches the vanishing line the projective
denominator w -> 0 and the mapped point runs off to infinity, then flips sign
behind the camera. `to_bev_meters` returns None for such points instead of a
garbage coordinate, and callers must treat None as "no measurement".
"""

import os
import json
import logging
from typing import Optional, Tuple

import numpy as np


class BEVProjector:
    def __init__(self, path: str, default_ppm: float, max_range_m: float = 300.0):
        self.H = np.eye(3, dtype=np.float64)
        self.ppm = float(default_ppm)
        self.max_range_m = float(max_range_m)
        self.calibrated = False
        self._w_sign = 1.0
        self._H_inv = None
        # Metric (survey-grade) calibration from lat/lng ground-control points.
        # When present it REPLACES the BEV-canvas homography: H then maps pixels
        # straight to East/North metres, so ppm is 1.0 and every downstream
        # threshold is in true physical units.
        self.geo = None
        self._load_metric(path)
        if self.geo is None:
            self._load(path)

    def _load_metric(self, path: str):
        from utils.geo_calib import load_gcp_calibration
        geo = load_gcp_calibration(path)
        if geo is None:
            return
        self.geo = geo
        self.H = geo.H
        self._H_inv = geo.H_inv
        self.ppm = 1.0                 # H already outputs metres
        self._w_sign = geo._w_sign
        self.calibrated = True
        logging.info("BEV in METRIC mode (lat/lng GCPs): RMSE=%.3f m P95=%.3f m",
                     geo.rmse_m, geo.p95_m)

    @property
    def metric(self) -> bool:
        """True when a survey-grade lat/lng calibration is in use."""
        return self.geo is not None

    @property
    def H_inv(self):
        if self._H_inv is None:
            try:
                self._H_inv = np.linalg.inv(self.H)
            except np.linalg.LinAlgError:
                self._H_inv = np.eye(3, dtype=np.float64)
        return self._H_inv

    def to_image_px(self, x_m: float, y_m: float):
        """Inverse of to_bev_meters: BEV-metre point -> image pixel (or None if
        it maps behind the camera). Used to overlay BEV forecasts on the frame."""
        bx, by = x_m * self.ppm, y_m * self.ppm
        v = self.H_inv @ np.array([bx, by, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-9:
            return None
        px, py = v[0] / v[2], v[1] / v[2]
        if not (np.isfinite(px) and np.isfinite(py)):
            return None
        return (float(px), float(py))

    def _load(self, path: str):
        if not os.path.isfile(path):
            logging.warning(
                f"BEV config '{path}' missing — using identity transform. "
                "All metric distances/speeds will be image-pixel based and WRONG; "
                "run bev_calibrator.py before trusting any output."
            )
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            H = np.array(cfg["homography_matrix"], dtype=np.float64)
            if H.shape != (3, 3) or abs(np.linalg.det(H)) < 1e-12:
                raise ValueError("homography_matrix must be an invertible 3x3 matrix")
            self.H = H
            self._H_inv = None
            ppm = float(cfg.get("pixels_per_meter", self.ppm))
            if ppm <= 0:
                raise ValueError(f"pixels_per_meter must be > 0, got {ppm}")
            self.ppm = ppm
            self.calibrated = True

            # Record the sign of the projective denominator over the calibrated
            # region: points on the far side of the horizon have the opposite
            # sign and must be rejected.
            src_pts = cfg.get("src_points")
            if src_pts:
                c = np.mean(np.asarray(src_pts, dtype=np.float64), axis=0)
                w = self.H[2, 0] * c[0] + self.H[2, 1] * c[1] + self.H[2, 2]
                self._w_sign = 1.0 if w >= 0 else -1.0
            else:
                self._w_sign = 1.0 if self.H[2, 2] >= 0 else -1.0

            logging.info(f"Loaded BEV config: ppm={self.ppm:.2f}")
        except Exception as e:
            logging.error(f"Failed to read BEV config: {e}. Using identity fallback.")

    def to_bev_meters(self, x: float, y: float) -> Optional[np.ndarray]:
        """Project an image ground-contact point to BEV metres.

        Returns None when the point is not projectable (at/above the horizon,
        or maps outside a sane range) — callers treat that as a missing
        measurement rather than propagating a corrupt coordinate.
        """
        w = self.H[2, 0] * x + self.H[2, 1] * y + self.H[2, 2]
        if w * self._w_sign < 1e-8:
            return None
        px = (self.H[0, 0] * x + self.H[0, 1] * y + self.H[0, 2]) / w
        py = (self.H[1, 0] * x + self.H[1, 1] * y + self.H[1, 2]) / w
        out = np.array([px, py], dtype=np.float64) / self.ppm
        if not np.all(np.isfinite(out)) or float(np.linalg.norm(out)) > self.max_range_m:
            return None
        return out

    def segment_length_m(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> Optional[float]:
        """Metric length of an image segment lying on the ground plane
        (e.g. the bottom edge of a bounding box). None if either end is
        unprojectable."""
        a = self.to_bev_meters(p1[0], p1[1])
        b = self.to_bev_meters(p2[0], p2[1])
        if a is None or b is None:
            return None
        return float(np.linalg.norm(b - a))
