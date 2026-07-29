"""§4.4 — Trajectory generation, kinematics, and short-horizon FORECAST.

Each tracked road user is a constant-velocity Kalman filter in the BEV plane
(metres), fed the bounding-box bottom-centre projected through the homography.
Kinematics (speed, acceleration, heading, yaw-rate) are derived with
least-squares slopes over a short window (robust to single-frame noise).

The forecast() method — the core of the predictive conflict logic — projects a
track's future path over the screening horizon:

  * CV   (constant velocity): straight-line, the proposal/RITSMS default.
  * CTRV (constant turn-rate + velocity): curved path, used when the track has
    a stable, significant yaw-rate (turning at an intersection) — this is where
    plain CV mispredicts a turning vehicle as going straight.

Acceleration is deliberately NOT extrapolated free-running (it is the noisiest
state and diverges over a 3 s horizon); it is reported and available for a
short speed nudge only. The forecast returns (t, x, y, theta) samples that the
conflict engine advances the oriented footprints along.
"""

import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List


def _lsq_slope(ts, vs) -> float:
    if len(ts) < 3:
        return 0.0
    t = np.asarray(ts, dtype=np.float64)
    v = np.asarray(vs, dtype=np.float64)
    t -= t.mean()
    denom = float(t @ t)
    if denom < 1e-9:
        return 0.0
    return float((t @ (v - v.mean())) / denom)


@dataclass
class TrackTrajectory:
    track_id: int
    cls_name: str
    cfg: object

    age: int = 0
    det_score: float = 0.0
    last_bbox: Optional[List[int]] = None
    last_img_point: Optional[tuple] = None       # bottom-centre (px)

    # BEV state
    bev_positions: deque = field(default_factory=lambda: deque(maxlen=30))
    raw_bev: Optional[np.ndarray] = None          # last accepted measurement
    speed: float = 0.0                            # m/s (BEV units)
    acc: float = 0.0                              # m/s^2 (signed)
    heading_deg: float = 0.0
    # Last heading observed while the track was genuinely moving. A vehicle
    # waiting at a signal keeps the heading it had on approach, so encounter
    # typing stays correct instead of using the ~0-speed noise direction.
    last_moving_heading_deg: Optional[float] = None
    yaw_rate: float = 0.0                         # rad/s
    direction_consistency: float = 0.0
    length_m: float = 4.5
    width_m: float = 1.8
    # Ground-plane unit vector pointing away from the camera at this track's
    # location; set by the caller from the homography. Used to anchor the
    # footprint at the vehicle's near face rather than its centre.
    away_dir: Optional[np.ndarray] = None

    # tracking-quality
    jump_count: int = 0
    n_updates: int = 0

    # Kalman internals [x, y, vx, vy]
    _mean: Optional[np.ndarray] = None
    _cov: Optional[np.ndarray] = None
    last_update_frame: int = -1
    _outlier_streak: int = 0
    _speed_hist: deque = field(default_factory=lambda: deque(maxlen=10))
    _head_hist: deque = field(default_factory=lambda: deque(maxlen=10))
    _unwrapped_head: Optional[float] = None

    def __post_init__(self):
        lw = self.cfg.class_size_lw_m.get(self.cls_name, (4.5, 1.8))
        self.length_m, self.width_m = float(lw[0]), float(lw[1])

    # ── properties ────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._mean is not None and self.n_updates >= 3 and len(self.bev_positions) >= 2

    @property
    def position_m(self) -> np.ndarray:
        return self._mean[:2].copy() if self._mean is not None else np.zeros(2)

    @property
    def velocity(self) -> np.ndarray:
        return self._mean[2:4].copy() if self._mean is not None else np.zeros(2)

    @property
    def radius_m(self) -> float:
        return 0.5 * self.width_m

    def footprint_center(self) -> np.ndarray:
        """Centre of the vehicle's ground footprint.

        The tracked point is the bbox bottom-centre, i.e. the ground contact of
        the vehicle's NEAR face (the rear bumper when seen from behind), not its
        centroid. Centring the L x W rectangle on it would push the footprint
        ~L/2 behind the real bumper, into the space a following or adjacent
        vehicle occupies -- which manufactured TTC=0 "contact" at IP33B.

        ``away_dir`` is the ground-plane direction leading away from the camera
        (supplied by the caller, which owns the homography), so the footprint is
        shifted from the near face towards the vehicle body.
        """
        p = self.position_m
        if self.away_dir is None:
            return p
        return p + self.away_dir * (0.5 * self.length_m)

    # ── Kalman ──────────────────────────────────────────────────────────────

    def _init_kf(self, z, frame_idx):
        c = self.cfg
        self._mean = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        sv = c.max_speed_mps / 4.0
        r = c.kf_measurement_noise_m ** 2
        self._cov = np.diag([r, r, sv * sv, sv * sv]).astype(np.float64)
        self.last_update_frame = frame_idx
        self._outlier_streak = 0
        self.bev_positions.clear()
        self.bev_positions.append(np.asarray(z, dtype=np.float64))
        self.raw_bev = np.asarray(z, dtype=np.float64)
        self._speed_hist.clear()
        self._head_hist.clear()
        self._unwrapped_head = None
        self.speed = self.acc = self.yaw_rate = 0.0
        self.direction_consistency = 0.0

    @staticmethod
    def _F(dt):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _Q(self, dt):
        s = self.cfg.kf_process_noise_accel ** 2
        q11 = dt ** 4 / 4.0
        q13 = dt ** 3 / 2.0
        q33 = dt ** 2
        return np.array([
            [q11, 0, q13, 0], [0, q11, 0, q13],
            [q13, 0, q33, 0], [0, q13, 0, q33],
        ], dtype=np.float64) * s

    def update(self, bev_pos, bbox, img_point, det_score, frame_idx, dt_s,
               measured_width_m=None, away_dir=None):
        """Advance the track one processed frame. bev_pos is metres (or None if
        the point was unprojectable — treated as a miss)."""
        c = self.cfg
        self.age += 1
        self.det_score = float(det_score)
        if bbox is not None:
            self.last_bbox = [int(v) for v in bbox]
        if img_point is not None:
            self.last_img_point = (int(img_point[0]), int(img_point[1]))
        if away_dir is not None:
            self.away_dir = np.asarray(away_dir, dtype=np.float64)
        if measured_width_m is not None and measured_width_m > 0:
            prior = c.class_size_lw_m.get(self.cls_name, (4.5, 1.8))[1]
            self.width_m = float(np.clip(measured_width_m, 0.5 * prior, 2.0 * prior))

        if bev_pos is None:
            return
        z = np.asarray(bev_pos, dtype=np.float64)

        if self._mean is None:
            self._init_kf(z, frame_idx)
            self.n_updates = 1
            return

        dt = max(dt_s, 1e-3)
        F = self._F(dt)
        mean_pred = F @ self._mean
        cov_pred = F @ self._cov @ F.T + self._Q(dt)

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        y = z - H @ mean_pred
        S = H @ cov_pred @ H.T + np.eye(2) * (c.kf_measurement_noise_m ** 2)
        try:
            d2 = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            d2 = float("inf")
        step = float(np.linalg.norm(z - self.bev_positions[-1]))
        implied_speed = step / dt
        if step > c.jitter_step_m:
            self.jump_count += 1

        if d2 > c.kf_gate_chi2 or implied_speed > c.max_speed_mps:
            self._outlier_streak += 1
            if self._outlier_streak >= c.kf_outlier_reset_frames:
                self._init_kf(z, frame_idx)
                self.n_updates = 1
                return
            self._mean, self._cov = mean_pred, cov_pred
            self.last_update_frame = frame_idx
            return

        K = cov_pred @ H.T @ np.linalg.inv(S)
        mean = mean_pred + K @ y
        cov = (np.eye(4) - K @ H) @ cov_pred
        v = mean[2:4]
        sp = float(np.linalg.norm(v))
        if sp > c.max_speed_mps:
            mean[2:4] = v * (c.max_speed_mps / sp)

        self._mean, self._cov = mean, 0.5 * (cov + cov.T)
        self.last_update_frame = frame_idx
        self._outlier_streak = 0
        self.n_updates += 1
        self.raw_bev = z
        self.bev_positions.append(mean[:2].copy())
        self._derive_kinematics(frame_idx * dt)

    def _derive_kinematics(self, t):
        c = self.cfg
        v = self.velocity
        self.speed = float(np.linalg.norm(v))
        if self.speed > c.min_speed_for_heading_mps:
            raw = math.atan2(v[1], v[0])
            if self._unwrapped_head is None:
                self._unwrapped_head = raw
            else:
                d = raw - (self._unwrapped_head % (2 * math.pi))
                d = (d + math.pi) % (2 * math.pi) - math.pi
                self._unwrapped_head += d
            self.heading_deg = math.degrees(raw)
            self._head_hist.append((t, self._unwrapped_head))
            # Remember it while the motion is fast enough to be trustworthy.
            if self.speed >= getattr(self.cfg, "min_moving_speed_mps", 1.0):
                self.last_moving_heading_deg = self.heading_deg
        self._speed_hist.append((t, self.speed))
        self.acc = _lsq_slope([s[0] for s in self._speed_hist], [s[1] for s in self._speed_hist])
        if self.speed > 1.0 and len(self._head_hist) >= 3:
            self.yaw_rate = _lsq_slope([h[0] for h in self._head_hist], [h[1] for h in self._head_hist])
        else:
            self.yaw_rate = 0.0
        self.direction_consistency = self._dir_consistency()

    def _dir_consistency(self) -> float:
        pts = list(self.bev_positions)[-7:]
        if len(pts) < 4:
            return 0.0
        angs = []
        for i in range(1, len(pts)):
            d = pts[i] - pts[i - 1]
            if np.linalg.norm(d) > 0.02:
                angs.append(math.atan2(d[1], d[0]))
        if len(angs) < 2:
            return 0.0
        return float(np.hypot(np.mean(np.cos(angs)), np.mean(np.sin(angs))))

    # ── FORECAST (CV / CTRV) ─────────────────────────────────────────────

    def forecast_mode(self) -> str:
        c = self.cfg
        yaw_dps = abs(math.degrees(self.yaw_rate))
        if (yaw_dps >= c.ctrv_min_yaw_rate_dps and
                self.speed >= c.ctrv_min_speed_mps and
                self.direction_consistency >= c.ctrv_min_dir_consistency):
            return "CTRV"
        return "CV"

    def forecast(self, horizon_s=None, dt_s=None) -> np.ndarray:
        """Predicted path as an array of rows [t, x, y, theta_rad] over the
        horizon. CV when going straight, CTRV (curved) when turning."""
        c = self.cfg
        horizon = c.forecast_horizon_s if horizon_s is None else horizon_s
        dt = c.forecast_dt_s if dt_s is None else dt_s
        ts = np.arange(0.0, horizon + 1e-9, dt)
        # Forecast the FOOTPRINT centre, so predicted rectangles sit on the
        # vehicle body rather than straddling the space behind its near face.
        p0 = self.footprint_center()
        v = self.velocity
        theta0 = math.atan2(v[1], v[0]) if self.speed > 1e-6 else math.radians(self.heading_deg)

        if self.forecast_mode() == "CTRV":
            w = self.yaw_rate
            sp = self.speed
            th = theta0 + w * ts
            x = p0[0] + (sp / w) * (np.sin(th) - math.sin(theta0))
            y = p0[1] + (sp / w) * (-np.cos(th) + math.cos(theta0))
            return np.column_stack([ts, x, y, th])
        # CV
        x = p0[0] + v[0] * ts
        y = p0[1] + v[1] * ts
        th = np.full_like(ts, theta0)
        return np.column_stack([ts, x, y, th])
