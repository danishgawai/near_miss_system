from dataclasses import dataclass, field
from typing import Dict, Set, List


@dataclass
class AppConfig:
    # I/O
    source_stream: str = "vehicle_crash.mp4"
    output_video_path: str = ""
    report_output_path: str = "near_miss_report.json"
    dashboard_output_path: str = "near_miss_dashboard.html"
    high_risk_frame_dir: str = "high_risk_frames"

    # Detection
    model_path: str = "models/yolo26n_int8_openvino_model/"
    model_conf: float = 0.2
    model_device: str = "cpu"
    filter_class_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    warmup_frames: int = 10

    # Tracker
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    track_max_box_ratio: float = 0.35
    min_track_age_for_risk: int = 6

    # BEV
    bev_config_path: str = "bev_config.json"
    default_pixels_per_meter: float = 80.0

    # Motion
    history_len: int = 30
    velocity_alpha: float = 0.45
    accel_alpha: float = 0.40
    hard_brake_mps2: float = -4.5
    hard_brake_min_speed_mps: float = 4.0
    use_optical_flow: bool = True

    # Near-miss
    proximity_gate_m: float = 18.0
    collision_radius_m: float = 1.8
    min_closing_speed_mps: float = 0.5
    ttc_max_eval_s: float = 4.0
    ttc_high_s: float = 1.0
    ttc_med_s: float = 1.8
    ttc_low_s: float = 3.0
    heading_conflict_deg: float = 65.0

    # Persistence
    incident_persist_frames: int = 5
    incident_cooldown_frames: int = 30

    # Runtime
    run_time_seconds: int = 3000

    # Classes
    class_mapping: Dict[int, str] = field(default_factory=lambda: {
        0: "pedestrian", 1: "bicycle", 2: "car",
        3: "motorcycle", 5: "bus", 7: "truck"
    })
    vulnerable_classes: Set[str] = field(default_factory=lambda: {"pedestrian", "bicycle", "motorcycle"})
    heavy_classes: Set[str] = field(default_factory=lambda: {"bus", "truck"})