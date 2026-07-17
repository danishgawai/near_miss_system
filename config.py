from dataclasses import dataclass, field
from typing import Dict, Set, List


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
    track_drop_after_missed: int = 8    # frames a TrackState survives unmatched

    # BEV
    bev_config_path: str = "bev_config.json"
    default_pixels_per_meter: float = 40.0
    bev_max_range_m: float = 300.0      # sanity bound on projected coordinates

    # Frame dimensions — set at runtime from video, used for border guard
    frame_width: int = 1280
    frame_height: int = 720
    frame_border_margin: int = 50

    # Motion estimation (constant-velocity Kalman filter in BEV metres)
    history_len: int = 30
    kf_process_noise_accel: float = 2.5   # m/s² — white-accel σ of the CV model
    kf_measurement_noise_m: float = 0.35  # m — BEV position measurement σ
    kf_gate_chi2: float = 9.21            # Mahalanobis gate, chi² 99% @ 2 dof
    kf_outlier_reset_frames: int = 5      # consecutive rejections before re-anchor
    max_speed_mps: float = 40.0           # hard physical ceiling
    min_speed_for_risk_mps: float = 0.8   # below this a heading is unreliable
    static_speed_mps: float = 0.5         # below this a track counts as stationary
    hard_brake_mps2: float = -4.5
    hard_brake_min_speed_mps: float = 6.0
    hard_brake_persist_frames: int = 3
    hard_brake_cooldown_frames: int = 30

    # Swerve
    swerve_lateral_acc_threshold: float = 3.5
    swerve_min_speed_mps: float = 3.0
    swerve_eval_frames: int = 5
    swerve_persist_frames: int = 3
    swerve_cooldown_frames: int = 30
    swerve_heading_delta_min_deg: float = 12.0

    # Near-miss geometry gates
    proximity_gate_m: float = 18.0
    min_closing_speed_mps: float = 0.5
    cpa_miss_distance_m: float = 4.0
    ttc_collision_buffer_m: float = 0.3   # safety margin added to r1 + r2

    # Object footprint priors — effective circular radius in metres.
    # Measured bottom-edge width (projected to BEV) is clipped to
    # [0.5, 2.0] × prior; the prior alone is the fallback.
    class_radius_m: Dict[str, float] = field(default_factory=lambda: {
        "pedestrian": 0.35, "bicycle": 0.7, "motorcycle": 0.8,
        "car": 1.4, "bus": 3.5, "truck": 3.0, "vehicle": 1.4,
    })

    # Lateral path-offset limits per scenario (metres)
    lat_limit_rear_end: float = 3.5
    lat_limit_merging: float = 5.0
    lat_limit_crossing: float = 7.0
    lat_limit_head_on: float = 4.0
    lat_limit_stationary: float = 2.5

    # TTC
    ttc_max_eval_s: float = 4.0
    ttc_default_gate_s: float = 1.5       # gate for class pairs not listed below

    # Per-class-pair TTC thresholds — binary gates only (not scoring refs)
    ttc_threshold_by_pair: Dict[str, float] = field(default_factory=lambda: {
        "motorcycle:motorcycle": 1.0,
        "car:car":               1.5,
        "car:motorcycle":        2.0,
        "car:pedestrian":        2.5,
        "car:bicycle":           2.5,
        "truck:pedestrian":      2.5,
        "bus:pedestrian":        2.5,
        "truck:bicycle":         2.5,
        "bus:bicycle":           2.5,
        "motorcycle:pedestrian": 2.0,
    })

    # ── Risk Index ────────────────────────────────────────────────────────
    # Weights (must sum to 1.0)
    ri_alpha: float = 0.35          # spatial proximity
    ri_beta: float = 0.45           # TTC urgency — most actionable
    ri_gamma: float = 0.20          # motion intensity (relative speed)

    # Reference denominators for normalisation — separate from gate thresholds
    ri_distance_ref: float = 6.0    # metres
    ri_ttc_ref: float = 2.0         # seconds
    ri_v_max: float = 15.0          # m/s — relative-speed reference

    # Risk thresholds
    ri_high_threshold: float = 0.65
    ri_medium_threshold: float = 0.40

    # Scenario heading thresholds (degrees)
    heading_rear_end_max_deg: float = 30.0
    heading_merging_max_deg: float = 75.0
    heading_crossing_max_deg: float = 140.0

    # Persistence / cooldown
    incident_persist_frames: int = 3
    incident_cooldown_frames: int = 15

    # Quality gates
    min_confidence_for_risk: float = 0.3       # detection score in [0, 1]
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
