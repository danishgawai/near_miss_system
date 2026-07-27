from dataclasses import dataclass, field
from typing import Dict, Set, List


@dataclass
class AppConfig:
    # I/O
    source_stream: str = "AD_intersection01.mp4.mp4"
    output_video_path: str = ""
    report_output_path: str = "near_miss_report.json"
    dashboard_output_path: str = "near_miss_dashboard.html"
    high_risk_frame_dir: str = "high_risk_frames"
    telemetry_csv_path: str = "near_miss_telemetry.csv"

    # Detection
    # Domain-trained road-user detector (YOLOv8s, merger8). ~3.7x the recall of
    # the generic COCO yolo26n on this intersection — the previous model was
    # blind to the far half of the scene, dropping tracks and whole conflicts.
    # NOTE its taxonomy differs from COCO (see class_mapping below).
    model_path: str = "models/yolov8s_merger8_exp1_int8_openvino_model/"
    model_conf: float = 0.2
    model_device: str = "cpu"
    filter_class_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
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

    # ── Collision detection ───────────────────────────────────────────────
    # A collision = contact + pre-impact approach + post-impact evidence.
    # Contact alone is NOT enough: adjacent-lane traffic and occlusion
    # artefacts produce footprint overlap without a crash.
    collision_overlap_m: float = 0.15          # edge distance counted as contact
    collision_min_track_age: int = 8           # lighter than near-miss age gate
    collision_watch_radius_m: float = 12.0     # record pair history within this
    collision_pre_window_frames: int = 10      # approach-evidence lookback
    collision_post_window_frames: int = 15     # frames to wait for impact evidence
    collision_min_closing_mps: float = 2.0     # required approach speed pre-contact
    collision_min_impact_rel_speed_mps: float = 2.5  # below this a "tap" is ignored
    collision_rel_speed_drop_ratio: float = 0.5      # momentum-exchange evidence
    collision_impact_decel_mps2: float = -3.0        # decel-spike evidence
    collision_stopped_speed_mps: float = 1.0         # both-at-rest evidence
    collision_persist_frames: int = 2          # contact frames before firing
    collision_cooldown_frames: int = 150       # a crash scene stays overlapped

    # Near-miss geometry gates
    proximity_gate_m: float = 18.0
    min_closing_speed_mps: float = 0.5
    cpa_miss_distance_m: float = 4.0
    ttc_collision_buffer_m: float = 0.3   # safety margin added to r1 + r2

    # Object footprint priors — effective circular radius in metres.
    # Measured bottom-edge width (projected to BEV) is clipped to
    # [0.5, 2.0] × prior; the prior alone is the fallback.
    # Effective circular radius ≈ half-WIDTH (RITSMS uses length×0.2 clamped
    # [0.3, 1.5]). Half-width — not half-length — avoids same-direction
    # adjacent-lane false positives in the TTC circle model. The runtime
    # refines this from the measured, projected bottom-edge width per track.
    class_radius_m: Dict[str, float] = field(default_factory=lambda: {
        "pedestrian": 0.35, "bicycle": 0.4, "motorcycle": 0.5,
        "car": 0.9, "bus": 1.5, "truck": 1.5, "vehicle": 0.9,
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

    # ── Detector backend ──────────────────────────────────────────────────
    detector_backend: str = "yolo"     # "yolo" | "rfdetr" (future — same infer() contract)
    post_nms_iou: float = 0.0          # >0 enables an explicit IoU de-dup pass pre-tracking

    # ── Site / calibration (proposal §4.2) ────────────────────────────────
    # ROI polygon + conflict zones live alongside the homography in this file.
    site_config_path: str = "bev_config.json"
    # BEV-only is the proposal baseline: TTC/PET are valid (scale-invariant /
    # time-based); absolute speed, accel, DRAC, Delta-V and *_M columns are
    # conditional until "metric_validated". A value in the site file overrides.
    calibration_confidence_default: str = "bev_only"  # | "metric_validated"
    fps_ceiling: float = 15.0          # effective-FPS cap for ingestion normalisation

    # ── Conflict measures (proposal §4.5–4.6) ─────────────────────────────
    ttc_screen_s: float = 3.0          # protocol TTC screening window
    nfb3_window_s: float = 0.5         # NFb3 look-back window
    nfb3_valid_fraction: float = 0.8   # ≥80% of window frames below ttc_screen_s
    pet_gate_vv_s: float = 3.0         # PET screening — vehicle-vehicle
    pet_gate_ped_s: float = 6.0        # PET screening — pedestrian/bicycle-vehicle
    drac_gate_mps2: float = 0.5        # DRAC screening threshold (rear-end)
    use_oriented_footprints: bool = False  # OBB/SAT TTC (needs stable heading)

    # ── RITSMS-aligned conflict engine ────────────────────────────────────
    # Severity levels are SAFE / WARNING / CRITICAL (proposal + RITSMS); only
    # WARNING and CRITICAL are published.
    min_publish_level: str = "WARNING"        # SAFE | WARNING | CRITICAL

    # TTC levels + closed-form solver (WARNING gate = ttc_screen_s above)
    ttc_critical_s: float = 1.5               # < this TTC -> CRITICAL
    ttc_horizon_s: float = 3.0                # constant-velocity forecast horizon
    ttc_min_rel_speed_mps: float = 0.5        # below this rel-speed the solve is skipped
    ttc_closing_eps_m2s: float = 0.1          # r·v < -eps (used by DRAC 2D)
    # A pair must be closing FASTER than this along the separation axis to be a
    # TTC conflict. Rejects queued / crawling same-direction traffic whose
    # footprint circles overlap without a genuine approach (dominant FP source).
    ttc_min_closing_mps: float = 2.0
    ttc_hysteresis_frames: int = 2            # consecutive CRITICAL frames before escalating
    # NFb3 temporal confirmation (TTC only) uses nfb3_window_s / nfb3_valid_fraction
    # above: window = round(fps*window_s); min = ceil(window*fraction).

    # PET (spatial-grid arrival-gap) — RITSMS parameters
    pet_cell_size_m: float = 1.5              # BEV occupancy-cell edge
    # >= this relative bearing = genuine crossing course. 30° (RITSMS) admits
    # near-parallel merges as false crossings in dense scenes; 60° keeps true
    # crossing/opposing conflicts (merging/side-swipe stay with the TTC path).
    pet_cross_heading_min_deg: float = 60.0
    pet_critical_s: float = 1.5               # < this PET -> CRITICAL
    pet_opposing_min_deg: float = 100.0       # >= this labels the event opposing_through
    pet_direction_min_speed_mps: float = 1.0  # min speed for a reliable bearing
    pet_max_gap_s: float = 6.0                # longest PET retained (ped window)

    # Per-pair emit dedupe (reporting): one incident per pair per window.
    nearmiss_dedupe_s: float = 3.0

    # Delta-V severity bands (m/s) — CONDITIONAL in BEV-only mode
    deltav_severe_rear_end_mps: float = 16.0
    deltav_severe_opposing_mps: float = 8.27
    deltav_severe_pedestrian_mps: float = 5.56

    # Class masses (kg) — Appendix B, for Delta-V. "truck" defaults to the
    # pickup/van figure (2250); override to 15000 for heavy commercial sites.
    class_mass_kg: Dict[str, float] = field(default_factory=lambda: {
        "pedestrian": 80.0, "bicycle": 90.0, "motorcycle": 250.0,
        "car": 2050.0, "truck": 2250.0, "bus": 18000.0, "vehicle": 2050.0,
    })

    # Runtime
    run_time_seconds: int = 3000

    # Classes — merger8 domain taxonomy (NOT COCO). Names feed class_radius_m,
    # class_mass_kg, vulnerable_classes, heavy_classes (all keyed by name).
    class_mapping: Dict[int, str] = field(default_factory=lambda: {
        0: "car", 1: "bus", 2: "truck",
        3: "motorcycle", 4: "pedestrian", 5: "bicycle"
    })
    vulnerable_classes: Set[str] = field(default_factory=lambda: {
        "pedestrian", "bicycle", "motorcycle"
    })
    heavy_classes: Set[str] = field(default_factory=lambda: {"bus", "truck"})
