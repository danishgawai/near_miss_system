"""Configuration for the RITSMS Near-Miss 2.0 pipeline.

Grounded in proposal4.pdf. All thresholds are here, none inline. Metric
quantities (speed m/s, DRAC, Delta-V, *_M coordinates) are reported as
CONDITIONAL until the site is metric-validated (calibration_confidence in the
site config); TTC and PET are valid in BEV-only mode (units cancel / time-based).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class Config:
    # ── Identity / IO ─────────────────────────────────────────────────────
    site_id: str = "IP86B"
    source: str = "IP86B_001.mp4"
    site_config_path: str = "bev_config.json"       # homography + ROI + zones
    output_dir: str = "ritsms_out"
    write_video: bool = True
    write_traces: bool = True        # per-pair TTC/PET traces for AC6 plots

    # ── Calibration guards ────────────────────────────────────────────────
    # Applying one camera's homography/ROI to another video yields plausible-
    # looking but geometrically wrong output, so it fails silently. These guards
    # make it fail loudly instead.
    strict_calibration: bool = True   # abort when the site config doesn't match the source
    # Coarse sanity thresholds only (see ritsms/calibration_check.py for why the
    # wander metric is informational and cannot validate a calibration).
    calib_stopped_speed_mps: float = 0.5   # below this a track is "stopped"
    calib_wander_warn_m: float = 2.0
    calib_wander_fail_m: float = 4.0
    calib_min_scene_span_m: float = 20.0   # a junction is tens of metres across
    calib_max_scene_span_m: float = 400.0

    # ── §4.1 Ingestion / FPS normalisation ────────────────────────────────
    fps_ceiling: float = 15.0        # effective processing-rate cap
    run_time_seconds: int = 100000   # wall-clock safety limit

    # ── §4.3 Detection ────────────────────────────────────────────────────
    detector_backend: str = "yolo"   # "yolo" | "rfdetr" (same infer() contract)
    # Default is the CPU OpenVINO INT8 export. For NVIDIA GPU, point model_path
    # at the .pt (models/yolov8s_merger8_exp1.pt) and set model_device to "0"/
    # "cuda" — requires a CUDA build of torch (see run.py --device/--model).
    model_path: str = "models/yolov8s_merger8_exp1.pt"
    model_conf: float = 0.2
    model_device: str = "cuda"          # "cpu" | "cuda" | "0"
    model_half: str = "fp16"          # FP16 (Pascal+ GPUs only; leave off on Maxwell)
    # Inference resolution. MEASURED on a GeForce 940MX (Maxwell): 640 -> 784 ms,
    # 480 -> 83 ms, 320 -> 50 ms per frame. The 640 cliff is ~9x worse than the
    # ~1.8x compute scaling predicts (low-bandwidth Maxwell memory), while inside
    # the ROI 480 retains ~104% of the detections of 640 — every detection lost
    # is distant background outside the analysed region. 480 is the sweet spot.
    model_imgsz: int = 640
    filter_class_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    warmup_frames: int = 10
    post_nms_iou: float = 0.0        # >0 enables an explicit IoU de-dup pass pre-tracking

    # merger8 taxonomy (NOT COCO)
    class_mapping: Dict[int, str] = field(default_factory=lambda: {
        0: "car", 1: "bus", 2: "truck", 3: "motorcycle", 4: "pedestrian", 5: "bicycle",
    })

    # ── §4.3 Tracking ─────────────────────────────────────────────────────
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    track_max_box_ratio: float = 0.35     # reject boxes larger than this fraction of frame
    track_drop_after_missed: int = 8

    # Tracking-quality monitor (AC2). Thresholds intentionally NOT frozen —
    # reported for the initial baseline; fragmentation is a GT-free proxy.
    frag_reassoc_dist_m: float = 3.0      # a new track this close to a dead one = fragment
    frag_reassoc_gap_frames: int = 15
    jitter_step_m: float = 6.0            # BEV step above this counts as a jump

    # ── §4.4 Trajectory / kinematics / forecast ───────────────────────────
    default_pixels_per_meter: float = 40.0
    bev_max_range_m: float = 300.0
    frame_border_margin: int = 50
    frame_width: int = 1280
    frame_height: int = 720

    history_len: int = 30
    smooth_window: int = 5                # moving-average window on BEV position
    min_track_age: int = 15               # frames before a track is analysed
    min_speed_for_heading_mps: float = 0.5
    max_speed_mps: float = 40.0

    # Kalman (constant-velocity, BEV metres)
    kf_process_noise_accel: float = 2.5
    kf_measurement_noise_m: float = 0.35
    kf_gate_chi2: float = 9.21
    kf_outlier_reset_frames: int = 5

    # Forecast (your trajectory-prediction idea)
    forecast_horizon_s: float = 3.0       # proposal screening horizon
    forecast_dt_s: float = 0.1            # forecast sampling step
    ctrv_min_yaw_rate_dps: float = 6.0    # |yaw| >= this -> use curved CTRV forecast
    ctrv_min_speed_mps: float = 1.5       # below this, heading/turn unreliable -> CV
    ctrv_min_dir_consistency: float = 0.4 # require a coherent recent heading for CTRV

    # ── §4.5 Conflict extraction ──────────────────────────────────────────
    proximity_gate_m: float = 25.0        # coarse pair pre-filter (centre distance)
    footprint_buffer_m: float = 0.3       # safety margin added to footprints
    # Motion state. The measures doc defines an interaction as "a pair of MOVING
    # objects simultaneously present in a scene", so a stopped vehicle is a
    # conflict TARGET but not an initiator. Rear-end into a signal queue is a
    # real and common conflict, so stopped vehicles are NOT excluded outright —
    # but at ~0 speed their velocity direction is pure noise, which at site
    # IP33B fabricated conflicts (incl. every "head_on") against vehicles
    # queued at the signal on the OPPOSITE carriageway.
    min_moving_speed_mps: float = 1.0     # below this a track is "stopped"
    # A stopped vehicle only counts as a target when the mover is genuinely on
    # course to hit it: ahead, and within this lateral corridor of its path.
    # Separates "approaching the back of my own queue" (real rear-end) from
    # "passing a queue stopped on the opposite lane" (not a conflict).
    stationary_lat_corridor_m: float = 2.5

    # Same-direction pairs: lane discipline. The rear-end TTC formula is defined
    # for vehicles "on the same line of travel" (Hayward 1972), so an
    # adjacent-lane pair violates its premise and must not be reported as a
    # rear-end. Measured at IP33B: 60% of reported rear-ends were >=2 m apart
    # laterally (i.e. a different lane), including 18 of the 23 TTC=0 events.
    lat_limit_rear_end_m: float = 2.0        # within this = same line of travel
    # Beyond that limit the pair is only a conflict if it is actually converging
    # laterally (a lane change / side-swipe). Steady parallel travel is not.
    # MEASURED at IP86B: a real lane change crosses ~3.5 m in 3-4 s = ~1.0 m/s
    # lateral, whereas a mere 7 deg heading error at 5 m/s fabricates 0.61 m/s.
    # At 0.3 the median flagged event was 0.72 m/s - barely above that noise
    # floor - so parallel travel with tracking jitter was being reported as a
    # side-swipe. 1.0 keeps genuine lane changes and rejects the jitter.
    sideswipe_min_lat_closing_mps: float = 1.0

    # TTC levels + solver
    ttc_warning_s: float = 3.0            # protocol screening window
    ttc_critical_s: float = 1.5
    ttc_min_closing_mps: float = 2.0      # reject queued/crawling non-approaching pairs
    use_oriented_footprints: bool = True  # SAT over predicted footprints (uses CTRV)

    # NFb3 validity (TTC): last window_s, >= fraction of frames below ttc_warning_s
    nfb3_window_s: float = 0.5
    nfb3_valid_fraction: float = 0.8

    # PET (hybrid: zone-based observational + predictive)
    pet_warning_vv_s: float = 3.0
    pet_warning_ped_s: float = 6.0
    pet_critical_s: float = 1.5
    pet_cell_size_m: float = 1.5          # grid cell for observational crossing PET
    pet_cross_min_deg: float = 60.0       # relative heading for a genuine crossing
    pet_max_gap_s: float = 6.0
    # PET is an ENCROACHMENT gap: one user clears the point, then the other
    # arrives. A gap of only a frame or two means they were there effectively
    # simultaneously — that is a collision-course (TTC) case, not a PET one, and
    # it fires constantly for a pedestrian standing beside passing traffic.
    # Below this floor the PET candidate is discarded.
    pet_min_s: float = 0.4

    # DRAC / Delta-V (reported; conditional on metric scale)
    drac_gate_mps2: float = 0.5
    deltav_severe_rear_end_mps: float = 16.0
    deltav_severe_opposing_mps: float = 8.27
    deltav_severe_pedestrian_mps: float = 5.56

    min_publish_level: str = "WARNING"    # SAFE | WARNING | CRITICAL
    nearmiss_dedupe_s: float = 3.0        # one conflict record per pair per window
    min_confidence: float = 0.3

    # ── Encounter taxonomy heading bands (degrees, §D5) ───────────────────
    heading_rear_end_max_deg: float = 30.0
    heading_merging_max_deg: float = 75.0
    heading_crossing_max_deg: float = 140.0
    # >= heading_crossing_max_deg -> head-on / opposing

    # ── Footprint priors (metres). radius ≈ half-width; L×W for oriented rects.
    #    Refined at runtime from the measured projected bottom-edge width.
    class_radius_m: Dict[str, float] = field(default_factory=lambda: {
        "pedestrian": 0.35, "bicycle": 0.4, "motorcycle": 0.5,
        "car": 0.9, "bus": 1.5, "truck": 1.5, "vehicle": 0.9,
    })
    class_size_lw_m: Dict[str, tuple] = field(default_factory=lambda: {
        "pedestrian": (0.6, 0.6), "bicycle": (1.8, 0.6), "motorcycle": (2.2, 0.8),
        "car": (4.5, 1.8), "bus": (12.0, 2.6), "truck": (10.0, 2.6), "vehicle": (4.5, 1.8),
    })

    # Appendix B masses (kg) for Delta-V
    class_mass_kg: Dict[str, float] = field(default_factory=lambda: {
        "pedestrian": 80.0, "bicycle": 90.0, "motorcycle": 250.0,
        "car": 2050.0, "truck": 2250.0, "bus": 18000.0, "vehicle": 2050.0,
    })

    vulnerable_classes: Set[str] = field(default_factory=lambda: {
        "pedestrian", "bicycle", "motorcycle",
    })
    ped_classes: Set[str] = field(default_factory=lambda: {"pedestrian", "bicycle"})
