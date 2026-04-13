import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Optional, Tuple


@dataclass
class TrackState:
    cls_name: str
    age: int = 0

    # BEV center trajectory (meters)
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    # Image bottom-center history (pixels)
    img_points: deque = field(default_factory=lambda: deque(maxlen=30))

    # Kinematics
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    acc: float = 0.0
    speed: float = 0.0

    # BEV key points (meters, rotation-aware)
    center_bev: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    front_bev: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    rear_bev: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    left_bev: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    right_bev: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    has_extent: bool = False

    # Raw image bbox and optional corridor tag
    bbox_xyxy: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    corridor_id: Any = None


class MotionEstimator:
    def __init__(self, fps: float, velocity_alpha: float, accel_alpha: float):
        self.fps = fps if fps > 0 else 30.0
        self.dt = 1.0 / self.fps
        self.v_alpha = float(velocity_alpha)
        self.a_alpha = float(accel_alpha)
        self.prev_gray = None

    # -----------------------------------------------------------------
    # Optical flow
    # -----------------------------------------------------------------
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

        # Keep reference fresh every call
        self.prev_gray = gray

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

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _safe_unit(v: np.ndarray) -> Tuple[np.ndarray, float]:
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32), 0.0
        return (v / n).astype(np.float32), n

    # -----------------------------------------------------------------
    # Lightweight center update (no extent)
    # -----------------------------------------------------------------
    def update_center(
        self,
        trk: TrackState,
        new_pos: np.ndarray,
        bbox_xyxy: Optional[np.ndarray] = None,
    ):
        trk.center_bev = new_pos.astype(np.float32)
        if bbox_xyxy is not None:
            trk.bbox_xyxy = bbox_xyxy.astype(np.float32)

    # -----------------------------------------------------------------
    # Rotation-aware extent (front/rear follow velocity heading)
    # -----------------------------------------------------------------
    def update_extent_from_bev_bbox(
        self,
        trk: TrackState,
        bev_bottom_left: np.ndarray,
        bev_bottom_right: np.ndarray,
        min_speed_for_heading: float = 0.7,
    ):
        bl = bev_bottom_left.astype(np.float32)
        br = bev_bottom_right.astype(np.float32)
        center = 0.5 * (bl + br)

        # NOTE: do NOT overwrite trk.center_bev here.
        # update_kinematics owns center_bev from the positions deque.

        width_vec = br - bl
        width_len = float(np.linalg.norm(width_vec))
        if width_len < 1e-5:
            trk.front_bev = trk.center_bev.copy()
            trk.rear_bev = trk.center_bev.copy()
            trk.left_bev = trk.center_bev.copy()
            trk.right_bev = trk.center_bev.copy()
            trk.has_extent = False
            return

        width_dir = width_vec / (width_len + 1e-6)

        # Heading: prefer velocity, fallback to trajectory, then width perpendicular
        v_norm = float(np.linalg.norm(trk.vel))
        if v_norm >= min_speed_for_heading:
            u = (trk.vel / (v_norm + 1e-6)).astype(np.float32)
        elif len(trk.positions) >= 2:
            d = trk.positions[-1] - trk.positions[-2]
            d_norm = float(np.linalg.norm(d))
            if d_norm > 1e-6:
                u = (d / (d_norm + 1e-6)).astype(np.float32)
            else:
                u = np.array([-width_dir[1], width_dir[0]], dtype=np.float32)
        else:
            u = np.array([-width_dir[1], width_dir[0]], dtype=np.float32)

        # Lateral axis orthogonal to heading
        n = np.array([-u[1], u[0]], dtype=np.float32)
        if float(np.dot(n, width_dir)) < 0:
            n = -n

        # Half-width from projected bottom edge
        half_w = max(0.35, 0.5 * width_len)

        # Class-aware half-length with safety clamp
        class_len_factor = {
            "pedestrian": 1.0,
            "bicycle": 1.4,
            "motorcycle": 1.6,
            "car": 2.2,
            "bus": 3.8,
            "truck": 4.2,
        }.get(trk.cls_name, 2.0)
        half_len = max(0.6, min(6.0, class_len_factor * half_w * 0.5))

        # Use kinematic center as anchor (not bottom-edge midpoint)
        c = trk.center_bev

        trk.front_bev = (c + half_len * u).astype(np.float32)
        trk.rear_bev = (c - half_len * u).astype(np.float32)
        trk.left_bev = (c - half_w * n).astype(np.float32)
        trk.right_bev = (c + half_w * n).astype(np.float32)
        trk.has_extent = True

    # -----------------------------------------------------------------
    # Kinematics (EMA velocity + acceleration)
    # -----------------------------------------------------------------
    def update_kinematics(
        self,
        trk: TrackState,
        new_pos: np.ndarray,
        flow_vel: Optional[np.ndarray] = None,
    ):
        new_pos = new_pos.astype(np.float32)

        if len(trk.positions) == 0:
            trk.positions.append(new_pos)
            trk.center_bev = new_pos.copy()
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

        trk.positions.append(new_pos)
        trk.center_bev = new_pos.copy()
        trk.vel = new_vel.astype(np.float32)
        trk.speed = new_speed
        trk.acc = float(new_acc)
        trk.age += 1