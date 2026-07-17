"""Motion state and estimation for tracked objects in BEV metric space.

Estimator design
----------------
- Per-track constant-velocity Kalman filter over state [x, y, vx, vy] in
  metres. dt is taken from the ACTUAL frame gap since the last accepted
  measurement, so occlusion gaps do not corrupt velocity.
- Outlier handling: a measurement failing the Mahalanobis gate (or implying a
  physically impossible speed) is skipped and the filter coasts on its
  prediction. Coasting inflates the covariance, so the gate widens over time
  and genuinely displaced tracks are re-accepted; after `outlier_reset_frames`
  consecutive rejections the filter re-anchors at the new position. This
  removes the failure mode where a track's BEV state froze permanently.
- Longitudinal acceleration = least-squares slope of speed(t) over a short
  window (robust to single-frame noise, unlike a chained-EMA derivative).
- Lateral acceleration = |v| * |yaw_rate| (centripetal), with yaw rate from
  the slope of the unwrapped heading — a physically meaningful swerve signal.
"""

import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List


def _lsq_slope(samples) -> float:
    """Least-squares slope of (t, value) samples. 0.0 if underdetermined."""
    if len(samples) < 3:
        return 0.0
    t = np.array([s[0] for s in samples], dtype=np.float64)
    v = np.array([s[1] for s in samples], dtype=np.float64)
    t -= t.mean()
    denom = float(np.dot(t, t))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(t, v - v.mean()) / denom)


@dataclass
class TrackState:
    track_id: int
    cls_name: str
    age: int = 0

    last_bbox: Optional[List[int]] = None
    det_score: float = 0.0          # detection confidence in [0, 1]
    bev_radius_m: float = 1.0       # effective footprint radius (metres)

    img_points: deque = field(default_factory=lambda: deque(maxlen=30))
    bev_positions: deque = field(default_factory=lambda: deque(maxlen=30))

    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    speed: float = 0.0              # m/s
    acc: float = 0.0                # longitudinal accel, m/s² (signed)
    heading_deg: float = 0.0
    yaw_rate: float = 0.0           # rad/s
    lateral_acc: float = 0.0        # m/s² (centripetal magnitude)

    lateral_acc_hist: deque = field(default_factory=lambda: deque(maxlen=15))
    heading_hist: deque = field(default_factory=lambda: deque(maxlen=15))

    motion_quality: float = 0.0         # smoothness of recent speed profile
    direction_consistency: float = 0.0  # circular resultant of step directions

    # Kalman internals
    kf_mean: Optional[np.ndarray] = None
    kf_cov: Optional[np.ndarray] = None
    last_update_frame: int = -1
    outlier_streak: int = 0
    _speed_samples: deque = field(default_factory=lambda: deque(maxlen=10))
    _heading_samples: deque = field(default_factory=lambda: deque(maxlen=10))
    _unwrapped_heading: Optional[float] = None

    @property
    def kinematics_valid(self) -> bool:
        return self.kf_mean is not None and len(self.bev_positions) >= 2


class MotionEstimator:
    _Hm = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    def __init__(
        self,
        fps: float,
        process_noise_accel: float = 2.5,
        measurement_noise_m: float = 0.35,
        gate_chi2: float = 9.21,          # chi-square 99%, 2 dof
        outlier_reset_frames: int = 5,
        max_speed_mps: float = 40.0,
        max_gap_s: float = 2.0,
    ):
        self.fps = fps if fps and fps > 0 else 30.0
        self.sigma_a = float(process_noise_accel)
        self.R = np.eye(2, dtype=np.float64) * float(measurement_noise_m) ** 2
        self.gate_chi2 = float(gate_chi2)
        self.outlier_reset_frames = int(outlier_reset_frames)
        self.max_speed_mps = float(max_speed_mps)
        self.max_gap_s = float(max_gap_s)

    # ── Kalman matrices ───────────────────────────────────────────────────

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # Discrete white-acceleration model for a CV filter
        q11 = dt ** 4 / 4.0
        q13 = dt ** 3 / 2.0
        q33 = dt ** 2
        Q = np.array([
            [q11, 0.0, q13, 0.0],
            [0.0, q11, 0.0, q13],
            [q13, 0.0, q33, 0.0],
            [0.0, q13, 0.0, q33],
        ], dtype=np.float64)
        return Q * self.sigma_a ** 2

    def _init_filter(self, trk: TrackState, z: np.ndarray, frame_idx: int):
        sv0 = self.max_speed_mps / 4.0
        trk.kf_mean = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        trk.kf_cov = np.diag([
            self.R[0, 0], self.R[1, 1], sv0 ** 2, sv0 ** 2
        ]).astype(np.float64)
        trk.last_update_frame = frame_idx
        trk.outlier_streak = 0
        trk.bev_positions.clear()
        trk.bev_positions.append(z.copy())
        trk._speed_samples.clear()
        trk._heading_samples.clear()
        trk._unwrapped_heading = None
        trk.vel = np.zeros(2, dtype=np.float64)
        trk.speed = 0.0
        trk.acc = 0.0
        trk.yaw_rate = 0.0
        trk.lateral_acc = 0.0
        trk.motion_quality = 0.4
        trk.direction_consistency = 0.0

    # ── quality metrics ───────────────────────────────────────────────────

    @staticmethod
    def _calc_motion_quality(trk: TrackState) -> float:
        """Scale-invariant smoothness: 1 / (1 + cv²) of recent speeds, where
        cv = std/mean. Unlike raw variance this does not penalise fast
        vehicles for their absolute speed."""
        if len(trk._speed_samples) < 4:
            return 0.4
        speeds = np.array([s[1] for s in trk._speed_samples], dtype=np.float64)
        mean = float(speeds.mean())
        cv = float(speeds.std()) / max(mean, 0.5)
        return float(np.clip(1.0 / (1.0 + cv * cv), 0.1, 1.0))

    @staticmethod
    def _calc_direction_consistency(trk: TrackState) -> float:
        """Circular resultant length of recent step directions in [0, 1]."""
        pts = list(trk.bev_positions)[-7:]
        if len(pts) < 4:
            return 0.0
        angs = []
        for i in range(1, len(pts)):
            d = pts[i] - pts[i - 1]
            if float(np.linalg.norm(d)) > 0.02:  # ignore sub-noise steps
                angs.append(float(np.arctan2(d[1], d[0])))
        if len(angs) < 2:
            return 0.0
        r = float(np.hypot(np.mean(np.cos(angs)), np.mean(np.sin(angs))))
        return float(np.clip(r, 0.0, 1.0))

    # ── main update ───────────────────────────────────────────────────────

    def update_track(self, trk: TrackState, bev_pos: Optional[np.ndarray], frame_idx: int):
        trk.age += 1
        if bev_pos is None:
            return  # unprojectable point — treat as a missed measurement

        z = np.asarray(bev_pos, dtype=np.float64)

        if trk.kf_mean is None:
            self._init_filter(trk, z, frame_idx)
            return

        gap = max(1, frame_idx - trk.last_update_frame)
        dt = min(gap / self.fps, self.max_gap_s)

        # Predict
        F = self._F(dt)
        mean_pred = F @ trk.kf_mean
        cov_pred = F @ trk.kf_cov @ F.T + self._Q(dt)

        # Gate
        y = z - self._Hm @ mean_pred
        S = self._Hm @ cov_pred @ self._Hm.T + self.R
        try:
            d2 = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            d2 = float("inf")
        implied_speed = float(np.linalg.norm(z - trk.bev_positions[-1])) / dt

        if d2 > self.gate_chi2 or implied_speed > self.max_speed_mps:
            trk.outlier_streak += 1
            if trk.outlier_streak >= self.outlier_reset_frames:
                # Sustained disagreement: the object really is elsewhere
                # (ID switch, occlusion exit) — re-anchor instead of freezing.
                self._init_filter(trk, z, frame_idx)
            else:
                # Coast on the prediction; growing covariance widens the gate
                # so a genuine jump is re-accepted within a few frames.
                trk.kf_mean = mean_pred
                trk.kf_cov = cov_pred
                trk.last_update_frame = frame_idx
            return

        # Update
        K = cov_pred @ self._Hm.T @ np.linalg.inv(S)
        mean = mean_pred + K @ y
        cov = (np.eye(4) - K @ self._Hm) @ cov_pred
        cov = 0.5 * (cov + cov.T)  # keep symmetric

        # Physical speed ceiling (numerical safety net)
        v = mean[2:4]
        s = float(np.linalg.norm(v))
        if s > self.max_speed_mps:
            mean[2:4] = v * (self.max_speed_mps / s)
            s = self.max_speed_mps

        trk.kf_mean = mean
        trk.kf_cov = cov
        trk.last_update_frame = frame_idx
        trk.outlier_streak = 0
        trk.bev_positions.append(mean[:2].copy())

        self._derive_kinematics(trk, frame_idx)

    def _derive_kinematics(self, trk: TrackState, frame_idx: int):
        t = frame_idx / self.fps
        trk.vel = trk.kf_mean[2:4].copy()
        trk.speed = float(np.linalg.norm(trk.vel))

        # Heading: hold last value below the noise floor, unwrap for yaw rate
        if trk.speed > 0.3:
            raw = float(np.arctan2(trk.vel[1], trk.vel[0]))
            if trk._unwrapped_heading is None:
                trk._unwrapped_heading = raw
            else:
                delta = raw - (trk._unwrapped_heading % (2.0 * np.pi))
                delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
                trk._unwrapped_heading += delta
            trk.heading_deg = float(np.degrees(raw))
            trk._heading_samples.append((t, trk._unwrapped_heading))

        # Longitudinal acceleration: slope of speed(t)
        trk._speed_samples.append((t, trk.speed))
        trk.acc = _lsq_slope(trk._speed_samples)

        # Lateral (centripetal) acceleration: |v| * |yaw rate|
        if trk.speed > 1.0 and len(trk._heading_samples) >= 3:
            trk.yaw_rate = _lsq_slope(trk._heading_samples)
            trk.lateral_acc = abs(trk.speed * trk.yaw_rate)
        else:
            trk.yaw_rate = 0.0
            trk.lateral_acc = 0.0

        trk.motion_quality = self._calc_motion_quality(trk)
        trk.direction_consistency = self._calc_direction_consistency(trk)
        trk.lateral_acc_hist.append(trk.lateral_acc)
        trk.heading_hist.append(trk.heading_deg)
