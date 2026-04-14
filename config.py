from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple


@dataclass
class AppConfig:
    # I/O
    source_stream: str = "traffic_light_video.mp4"
    output_video_path: str = ""
    report_output_path: str = "near_miss_report.json"
    dashboard_output_path: str = "near_miss_dashboard.html"
    high_risk_frame_dir: str = "high_risk_frames"
    telemetry_csv_path: str = "near_miss_telemetry.csv"

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
    min_track_age_for_risk: int = 15

    # BEV
    bev_config_path: str = "bev_config.json"
    default_pixels_per_meter: float = 40.0

    # Frame dimensions — set at runtime from video, used for border guard
    frame_width: int = 1280
    frame_height: int = 720
    frame_border_margin: int = 50

    # Motion
    history_len: int = 30
    velocity_alpha: float = 0.45
    accel_alpha: float = 0.40
    max_speed_mps: float = 40.0
    min_speed_for_risk_mps: float = 0.8
    hard_brake_mps2: float = -4.5
    hard_brake_min_speed_mps: float = 6.0

    # Swerve
    swerve_lateral_acc_threshold: float = 3.5
    swerve_min_speed_mps: float = 3.0
    swerve_eval_frames: int = 5
    swerve_persist_frames: int = 3
    swerve_heading_delta_min_deg: float = 12.0

    # Near-miss geometry gates
    proximity_gate_m: float = 18.0
    min_closing_speed_mps: float = 0.5
    cpa_miss_distance_m: float = 4.0

    # Lateral separation limits per scenario (metres)
    lat_limit_rear_end: float = 3.5
    lat_limit_merging: float = 5.0
    lat_limit_crossing: float = 7.0
    lat_limit_head_on: float = 4.0

    # TTC
    ttc_max_eval_s: float = 4.0
    ttc_low_s: float = 3.0

    # Per-class-pair TTC thresholds — used as binary gates only (not scoring refs)
    ttc_threshold_by_pair: Dict[str, float] = field(default_factory=lambda: {
        "motorcycle:motorcycle": 1.0,
        "car:car":               1.5,
        "car:motorcycle":        2.0,
        "motorcycle:car":        2.0,
        "car:pedestrian":        2.5,
        "pedestrian:car":        2.5,
        "truck:pedestrian":      2.5,
        "pedestrian:truck":      2.5,
        "bus:pedestrian":        2.5,
        "pedestrian:bus":        2.5,
    })

    # ── Risk Index ────────────────────────────────────────────────────────
    # Weights (must sum to 1.0)
    ri_alpha: float = 0.35          # spatial proximity
    ri_beta: float  = 0.45          # TTC urgency  ← increased, most actionable
    ri_gamma: float = 0.20          # motion intensity

    # Reference denominators for normalisation — separate from gate thresholds
    ri_distance_ref: float = 6.0    # metres — danger zone reference distance
    ri_ttc_ref: float = 2.0         # seconds — scoring reference TTC
    ri_v_max: float = 15.0          # m/s — speed-diff reference

    # Risk thresholds
    ri_high_threshold: float = 0.65
    ri_medium_threshold: float = 0.40

    # Scenario heading thresholds (degrees)
    heading_rear_end_max_deg: float = 30.0
    heading_merging_max_deg: float  = 75.0
    heading_crossing_max_deg: float = 140.0

    # Persistence / cooldown
    incident_persist_frames: int = 3
    incident_cooldown_frames: int = 15

    # Quality gates
    min_confidence_for_risk: float = 0.3
    direction_consistency_min: float = 0.25

    # Runtime
    run_time_seconds: int = 3000

    # Classes
    class_mapping: Dict[int, str] = field(default_factory=lambda: {
        0: "pedestrian", 1: "bicycle", 2: "car",
        3: "motorcycle", 5: "bus", 7: "truck"
    })
    vulnerable_classes: Set[str] = field(default_factory=lambda: {
        "pedestrian", "bicycle", "motorcycle"
    })
    heavy_classes: Set[str] = field(default_factory=lambda: {"bus", "truck"})