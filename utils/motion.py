"""Motion state and estimation for tracked objects in BEV space."""

import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List


@dataclass
class TrackState:
    track_id: int
    cls_name: str
    age: int = 0

    img_points: deque = field(default_factory=lambda: deque(maxlen=30))
    bev_positions: deque = field(default_factory=lambda: deque(maxlen=30))

    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    acc_vec: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    speed: float = 0.0
    acc: float = 0.0
    heading_deg: float = 0.0
    lateral_acc: float = 0.0
    lateral_acc_hist: deque = field(default_factory=lambda: deque(maxlen=10))
    heading_hist: deque = field(default_factory=lambda: deque(maxlen=10))
    confidence: float = 0.0
    direction_consistency: float = 0.0

    last_bbox: Optional[List[int]] = None


class MotionEstimator:
    def __init__(
        self,
        fps: float,
        velocity_alpha: float = 0.45,
        accel_alpha: float = 0.35,
        max_speed_mps: float = 40.0,
    ):
        self.fps = fps if fps > 0 else 30.0
        self.dt = 1.0 / self.fps
        self.v_alpha = float(velocity_alpha)
        self.a_alpha = float(accel_alpha)
        self.max_speed_mps = float(max_speed_mps)   # hard physical ceiling

    @staticmethod
    def _safe_norm(v: np.ndarray, eps: float = 1e-6):
        n = float(np.linalg.norm(v))
        return v / (n + eps), n

    def _calc_confidence(self, trk: TrackState) -> float:
        if len(trk.bev_positions) < 5:
            return 0.4
        recent = list(trk.bev_positions)[-8:]
        if len(recent) < 4:
            return 0.5
        speeds = []
        for i in range(1, len(recent)):
            dv = recent[i] - recent[i - 1]
            speeds.append(float(np.linalg.norm(dv)) * self.fps)
        if len(speeds) < 2:
            return 0.5
        var = float(np.var(speeds))
        return float(np.clip(1.0 / (1.0 + var), 0.1, 1.0))

    def _calc_direction_consistency(self, trk: TrackState) -> float:
        pts = list(trk.bev_positions)[-7:]
        if len(pts) < 4:
            return 0.0
        vecs = [pts[i] - pts[i - 1] for i in range(1, len(pts))]
        angs = [
            float(np.arctan2(v[1], v[0]))
            for v in vecs
            if float(np.linalg.norm(v)) > 0.01
        ]
        if len(angs) < 2:
            return 0.0
        mean_cos = float(np.mean([np.cos(a) for a in angs]))
        mean_sin = float(np.mean([np.sin(a) for a in angs]))
        r = np.sqrt(mean_cos ** 2 + mean_sin ** 2)
        return float(np.clip(r, 0.0, 1.0))

    def update_track(self, trk: TrackState, bev_pos: np.ndarray):
        bev_pos = bev_pos.astype(np.float32)

        if len(trk.bev_positions) == 0:
            trk.bev_positions.append(bev_pos)
            trk.vel = np.zeros(2, dtype=np.float32)
            trk.acc_vec = np.zeros(2, dtype=np.float32)
            trk.speed = 0.0
            trk.acc = 0.0
            trk.heading_deg = 0.0
            trk.lateral_acc = 0.0
            trk.confidence = 0.4
            trk.direction_consistency = 0.0
            trk.age += 1
            return

        prev_pos = trk.bev_positions[-1]
        raw_vel = (bev_pos - prev_pos) / self.dt

        # ── Velocity spike rejection ────────────────────────────────────────
        # If the raw displacement implies a physically impossible speed,
        # discard this position update entirely (bad BEV point / occlusion).
        if float(np.linalg.norm(raw_vel)) > self.max_speed_mps:
            trk.age += 1
            return

        # Adaptive alpha: young tracks react faster to build up estimate,
        # but are guarded by the spike cap above.
        a_v = min(0.75, self.v_alpha + 0.2 / (trk.age + 1))
        prev_vel = trk.vel.copy()
        new_vel = a_v * raw_vel + (1.0 - a_v) * trk.vel
        new_speed = float(np.linalg.norm(new_vel))

        raw_acc_vec = (new_vel - prev_vel) / self.dt
        new_acc_vec = self.a_alpha * raw_acc_vec + (1.0 - self.a_alpha) * trk.acc_vec

        prev_speed = trk.speed
        raw_acc = (new_speed - prev_speed) / self.dt
        new_acc = self.a_alpha * raw_acc + (1.0 - self.a_alpha) * trk.acc

        heading = (
            float(np.degrees(np.arctan2(new_vel[1], new_vel[0])))
            if new_speed > 0.05
            else trk.heading_deg
        )

        lateral_acc = 0.0
        if new_speed > 1.0:
            h, _ = self._safe_norm(new_vel)
            lateral_dir = np.array([-h[1], h[0]], dtype=np.float32)
            lateral_acc = abs(float(np.dot(new_acc_vec, lateral_dir)))

        trk.bev_positions.append(bev_pos)
        trk.vel = new_vel.astype(np.float32)
        trk.acc_vec = new_acc_vec.astype(np.float32)
        trk.speed = new_speed
        trk.acc = float(new_acc)
        trk.heading_deg = heading
        trk.lateral_acc = lateral_acc
        trk.confidence = self._calc_confidence(trk)
        trk.direction_consistency = self._calc_direction_consistency(trk)
        trk.lateral_acc_hist.append(lateral_acc)
        trk.heading_hist.append(heading)
        trk.age += 1
