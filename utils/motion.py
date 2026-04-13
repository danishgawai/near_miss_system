import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Tuple, Optional


@dataclass
class TrackState:
    cls_name: str
    age: int = 0
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    img_points: deque = field(default_factory=lambda: deque(maxlen=30))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    acc: float = 0.0
    speed: float = 0.0


class MotionEstimator:
    def __init__(self, fps: float, velocity_alpha: float, accel_alpha: float):
        self.fps = fps if fps > 0 else 30.0
        self.dt = 1.0 / self.fps
        self.v_alpha = float(velocity_alpha)
        self.a_alpha = float(accel_alpha)
        self.prev_gray = None

    def update_optical_flow_reference(self, frame_bgr):
        self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def refine_velocity_with_flow(
        self,
        curr_bgr,
        img_prev: Tuple[int, int],
        img_curr: Tuple[int, int],
        bev_delta_m: np.ndarray,
    ) -> np.ndarray:
        base_vel = bev_delta_m / self.dt
        if self.prev_gray is None:
            return base_vel.astype(np.float32)

        gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        p0 = np.array([[img_prev]], dtype=np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            p0,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        if p1 is None or st is None or st[0][0] != 1:
            return base_vel.astype(np.float32)

        flow = (p1[0][0] - p0[0][0]).astype(np.float32)
        flow_norm = float(np.linalg.norm(flow))
        base_norm = float(np.linalg.norm(base_vel))
        if flow_norm < 0.5 or base_norm < 1e-6:
            return base_vel.astype(np.float32)

        flow_dir = flow / (flow_norm + 1e-6)
        base_dir = base_vel / (base_norm + 1e-6)
        align = float(np.clip(np.dot(flow_dir, base_dir), -1.0, 1.0))

        if align <= 0.0:
            return base_vel.astype(np.float32)

        gain = min(0.15, 0.15 * align)
        refined = base_vel * (1.0 + gain)
        return refined.astype(np.float32)

    def update_kinematics(self, trk: TrackState, new_pos: np.ndarray, flow_vel: Optional[np.ndarray] = None):
        if len(trk.positions) == 0:
            trk.positions.append(new_pos.astype(np.float32))
            trk.vel = np.zeros(2, dtype=np.float32)
            trk.acc = 0.0
            trk.speed = 0.0
            trk.age += 1
            return

        prev_pos = trk.positions[-1]
        raw_vel = (new_pos - prev_pos) / self.dt
        if flow_vel is not None:
            raw_vel = 0.7 * raw_vel + 0.3 * flow_vel

        new_vel = self.v_alpha * raw_vel + (1.0 - self.v_alpha) * trk.vel
        new_speed = float(np.linalg.norm(new_vel))
        raw_acc = (new_speed - trk.speed) / self.dt
        new_acc = self.a_alpha * raw_acc + (1.0 - self.a_alpha) * trk.acc

        trk.positions.append(new_pos.astype(np.float32))
        trk.vel = new_vel.astype(np.float32)
        trk.speed = new_speed
        trk.acc = float(new_acc)
        trk.age += 1