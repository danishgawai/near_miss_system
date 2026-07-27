"""Site geometry for conflict analysis: the ROI polygon and named conflict
zones, plus calibration-confidence metadata. Read from the same JSON as the
homography (bev_config.json by default).

Coordinate frame
----------------
Polygons are expressed in the BEV analysis plane — the same
internally-consistent transformed coordinates that MotionEstimator produces
(``TrackState.bev_positions``). In the proposal's BEV-only baseline (§4.2)
this plane is not guaranteed metric, but PET (a time gap) and TTC (units
cancel) remain valid there, so ROI/zone geometry defined in this plane is
sufficient for conflict screening.

If no ROI is defined the whole plane is analysed; if no zones are defined
PET simply has nothing to key off. Both degrade gracefully so the pipeline
runs before a site has been fully configured.
"""

import os
import json
import logging
from typing import List, Sequence

import numpy as np
import cv2


def _as_contour(points: Sequence) -> np.ndarray:
    """(N,2) polygon -> cv2 contour of shape (N,1,2) float32."""
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


class ConflictZone:
    """A named polygon in the BEV plane used for PET entry/exit logic."""

    def __init__(self, zone_id: str, polygon: Sequence):
        self.id = str(zone_id)
        self.polygon = _as_contour(polygon)

    def contains(self, xy) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(xy[0]), float(xy[1])), False) >= 0


class SiteConfig:
    def __init__(self, path: str, default_confidence: str = "bev_only"):
        self.path = path
        self.calibration_confidence = default_confidence
        self.reprojection_rmse_m = None      # populated only when metric-validated
        self.reprojection_p95_m = None
        self.roi_polygon = None              # BEV-plane ROI contour (N,1,2) or None
        # Image-space processing ROI (pixels). When enabled, the pipeline only
        # analyses detections whose ground-contact point falls inside it.
        self.roi_enabled = False
        self.roi_polygon_img = None          # cv2 contour (N,1,2) in pixels or None
        self.zones: List[ConflictZone] = []
        self._load(path)

    # ── metadata ──────────────────────────────────────────────────────────

    @property
    def metric_confident(self) -> bool:
        """True only when survey-validated scale exists — gates whether metric
        columns (speed m/s, DRAC, Delta-V, *_M positions) are reported."""
        return self.calibration_confidence == "metric_validated"

    @property
    def bev_roi_enabled(self) -> bool:
        """Whether a BEV-plane analysis ROI is defined (distinct from the
        image-space processing ROI in ``roi_enabled``)."""
        return self.roi_polygon is not None

    # ── loading ─────────────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        if not os.path.isfile(path):
            logging.warning(
                f"Site config '{path}' missing — ROI/zones disabled, "
                f"calibration_confidence='{self.calibration_confidence}'."
            )
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            logging.error(f"Failed to read site config '{path}': {e}")
            return

        self.calibration_confidence = str(
            cfg.get("calibration_confidence", self.calibration_confidence)
        )
        self.reprojection_rmse_m = cfg.get("reprojection_rmse_m")
        self.reprojection_p95_m = cfg.get("reprojection_p95_m")

        roi = cfg.get("roi_polygon_bev_m")
        if roi and len(roi) >= 3:
            self.roi_polygon = _as_contour(roi)
        elif roi:
            logging.warning("roi_polygon_bev_m has < 3 points — ignored.")

        # Image-space processing ROI (from the web calibrator).
        roi_img = cfg.get("roi_polygon_img")
        if roi_img and len(roi_img) >= 3:
            self.roi_polygon_img = _as_contour(roi_img)
            self.roi_enabled = bool(cfg.get("roi_enabled", True))
        else:
            self.roi_enabled = False

        for z in cfg.get("conflict_zones", []) or []:
            poly = z.get("polygon_bev_m")
            if poly and len(poly) >= 3:
                self.zones.append(
                    ConflictZone(z.get("id", f"zone_{len(self.zones)}"), poly)
                )
            else:
                logging.warning(f"Conflict zone {z.get('id')} skipped — needs >= 3 points.")

        logging.info(
            f"Site config: confidence={self.calibration_confidence}, "
            f"image_roi={'on' if self.roi_enabled else 'off (full frame)'}, "
            f"bev_roi={'yes' if self.roi_polygon is not None else 'no'}, "
            f"zones={len(self.zones)}"
        )

    # ── queries ─────────────────────────────────────────────────────────────

    def point_in_roi(self, xy) -> bool:
        """True if the BEV point is inside the BEV-plane ROI (or none defined)."""
        if self.roi_polygon is None:
            return True
        return cv2.pointPolygonTest(self.roi_polygon, (float(xy[0]), float(xy[1])), False) >= 0

    def point_in_image_roi(self, x: float, y: float) -> bool:
        """True if the image-space point is inside the processing ROI. Always
        True when the ROI is disabled (full-frame processing)."""
        if not self.roi_enabled or self.roi_polygon_img is None:
            return True
        return cv2.pointPolygonTest(self.roi_polygon_img, (float(x), float(y)), False) >= 0

    def zones_containing(self, xy) -> List[str]:
        """IDs of every conflict zone containing the BEV point."""
        return [z.id for z in self.zones if z.contains(xy)]
