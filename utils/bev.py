import os
import json
import logging
import numpy as np
import cv2


class BEVProjector:
    def __init__(self, path: str, default_ppm: float):
        self.H = np.eye(3, dtype=np.float32)
        self.ppm = float(default_ppm)
        self._load(path)

    def _load(self, path: str):
        if not os.path.isfile(path):
            logging.warning(f"BEV config '{path}' missing. Using identity transform.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.H = np.array(cfg["homography_matrix"], dtype=np.float32)
            self.ppm = float(cfg.get("pixels_per_meter", self.ppm))
            if self.ppm <= 0:
                self.ppm = 80.0
            logging.info(f"Loaded BEV config with ppm={self.ppm:.2f}")
        except Exception as e:
            logging.error(f"Failed to read BEV config: {e}. Using identity fallback.")

    def to_bev_meters(self, x: float, y: float) -> np.ndarray:
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)[0][0]
        return out / self.ppm