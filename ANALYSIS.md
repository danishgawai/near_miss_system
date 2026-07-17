# Near-Miss System — Technical Review & Rewrite Notes

Expert review of the original pipeline (computer vision + mathematics), the
defects found, and the corrections applied. Every item below was verified
against the code, and the fixes are covered by the numerical checks in the
test script (Kalman convergence, TTC quadratic, crossing/stationary
scenarios, brake persistence, state pruning).

---

## 1. Critical correctness bugs (silently wrong output)

### 1.1 Detection scores fed to ByteTrack at 100× scale
`yolo_infer.py` emitted `int(score * 100)` (0–100) while ByteTrack assumes
scores in [0, 1]:

- `scores > track_thresh (0.5)` was **always true** → the second (low-score
  recovery) association stage — the entire point of BYTE — never executed.
- `fuse_score()` computes `cost = 1 − iou·score`; with score ≈ 90 the cost
  went to ≈ −80, so the assignment threshold (`0.8`) became meaningless and
  any overlap matched.
- `min_confidence_for_risk = 0.3` was never compared against a real
  detection score (see 1.4).

**Fix:** scores stay `float` in [0, 1] end-to-end.

### 1.2 BEV projection used the bbox centre, not the ground point
A planar homography maps **ground-plane** points only. The bbox centre sits
roughly half the object's height above the ground; projecting it through H
lands metres away from the true position, with a bias that grows with object
height and camera obliqueness (a bus was systematically displaced more than a
pedestrian). Every downstream quantity — distance, speed, TTC, RI — inherited
this bias. (The README already claimed bottom-centre was used; the code
didn't.)

**Fix:** project the bottom-centre `((x1+x2)/2, y2)`.

### 1.3 Motion intensity used |s₁ − s₂| instead of |v₁ − v₂|
The risk term `R_v` used the difference of speed *magnitudes*. For two
vehicles meeting head-on at 10 m/s each, `|10 − 10| = 0` → zero motion risk
for the most violent possible conflict. Same for 90° crossings. The correct
scalar is the **relative velocity magnitude** ‖v₁ − v₂‖ (14.1 m/s for the
crossing, 20 m/s for the head-on).

### 1.4 "Confidence" gate compared the wrong quantity
The tracker's detection score was never stored in `TrackState`; the field
named `confidence` was actually `1/(1+var(speeds))` — a motion-smoothness
statistic whose *units* are (m/s)², so it also penalised fast vehicles for
being fast. **Fix:** `det_score` (real detector confidence) is stored and
gated; smoothness became `motion_quality = 1/(1+cv²)` using the
coefficient of variation (scale-invariant).

### 1.5 Moving-vehicle vs. stationary-object conflicts were unreachable
Gate 6 passed if *either* track moved, but gate 7 (heading delta) required
**both** speeds ≥ 0.8 m/s, otherwise the sentinel `999.0` rejected the pair.
A car bearing down on a standing pedestrian — the highest-value near-miss
category for a CCTV product — could **never** fire. **Fix:** explicit
`stationary` scenario: the geometry is evaluated in the mover's path frame
(target must be ahead and within `lat_limit_stationary` of the travel axis),
and CPA/TTC remain well-defined since relative velocity reduces to the
mover's velocity.

### 1.6 Lateral-separation "equation" didn't measure lateral separation
Old formula: `lat_sep = min(|v₁·û_lat|, |v₂·û_lat|) × t_close`, where
`û_lat ⊥` the *separation* axis. This is neither the current lateral offset
nor the relative lateral drift — it's the projected sideways travel of the
slower-drifting vehicle. Consequence: two vehicles on a genuine 90°
collision course (each ~7 m/s of "lateral" velocity relative to the
separation axis, t_close ≈ 2 s) produced `lat_sep ≈ 14 m` → **rejected by the
7 m crossing gate precisely when they were about to collide.**

**Fix:** lateral offset is now the standard path-frame quantity
`lat_i = |ĥ_i × (p_j − p_i)|` (perpendicular distance of the other object
from track *i*'s travel axis), gated per scenario. Trajectory convergence is
handled where it belongs — in the CPA/TTC gate.

### 1.7 Velocity-spike rejection froze tracks permanently
When a raw displacement implied speed > 40 m/s, the old estimator discarded
the position **and kept the stale one as the reference**, so the next frame's
displacement was even larger → rejected again → the BEV state froze forever
while the bbox kept moving (phantom geometry for every pair involving that
track). Also `dt` was hard-coded to `1/fps`, so any occlusion gap corrupted
velocity by the gap factor.

**Fix (motion rewrite):** constant-velocity Kalman filter with
- `dt` = actual frame gap,
- Mahalanobis gating (χ², 99 %, 2 dof) — a rejected measurement lets the
  filter *coast*, which inflates covariance and widens the gate so genuine
  jumps are re-accepted within a few frames,
- hard re-anchor after N consecutive rejections (ID switch / occlusion exit).

### 1.8 `pixels_per_meter` applied in the wrong coordinate space
`_approx_bev_half_diag` divided **image**-pixel bbox dimensions by the BEV
canvas's `pixels_per_meter`. Image scale varies with depth (that is what the
homography corrects); the two spaces are dimensionally incompatible. It also
used `default_pixels_per_meter` (40) even when a calibrated value existed.
**Fix:** the bbox bottom edge (a ground-plane segment) is projected through H
— its metric length is well-defined — then `r = clip(width/2, [0.5, 2.0] ×
class prior)`, with the class prior as fallback.

### 1.9 No horizon guard on the homography
Points approaching the camera's vanishing line drive the projective
denominator `w → 0`: coordinates blow up and then flip sign (behind the
camera). The old code returned these as valid metres. **Fix:** the projector
returns `None` for `w` on the wrong side of the horizon or beyond a range
bound, and the estimator treats `None` as a missed measurement.

---

## 2. Event-logic defects

| Defect | Consequence | Fix |
|---|---|---|
| Hard braking had no persistence/cooldown | one braking manoeuvre → up to ~30 duplicate incidents (one per frame) | persist ≥ 3 frames + 30-frame cooldown → exactly one event |
| Persistence hard-reset on any gate failure | one flickered frame erased accumulated evidence | leaky counter: soft-gate failures decay by 1; structural failures (age, missing data, border) still reset |
| `pair_persist` / `swerve_persist` never pruned | unbounded memory growth on RTSP streams | `prune(live_ids)` called every frame |
| `print(output_stracks)` every frame in the tracker | log spam, ~ms/frame wasted | removed |
| TTC gate fell back to `ttc_low_s = 3.0` for unlisted pairs | car:bus (3.0 s) was gated *more leniently* than car:car (1.5 s) | explicit `ttc_default_gate_s = 1.5` + completed pair table (reverse pairs looked up automatically) |
| Acceleration = EMA(d/dt(EMA(‖EMA(v)‖))) | triple-smoothed, lagged, correlated noise under a −4.5 m/s² threshold | least-squares slope of speed(t) over a short window |
| Swerve lateral accel = `|a⃗_ema · l̂|` | inherited the same lag chain | centripetal form `a_lat = |v|·|ψ̇|`, yaw rate from regression of unwrapped heading |
| `TelemetryLogger` dead code | README promised CSV telemetry; none was written | wired into the main loop (per-track kinematics + max RI per frame) |
| Wall-clock-only timestamps | offline video incidents carried processing time, not video time | `video_time_s = frame/fps` added to every incident |
| `fps = cap.get(...) or 30` | RTSP sources often report 0/NaN/90000 — every speed and TTC scales linearly with fps | validated range [1, 120] with explicit warning |
| Actor IDs parsed back out of display strings in visualization | brittle string surgery | incidents carry `actor_1_id` / `actor_2_id` |

---

## 3. Mathematical upgrades

### 3.1 TTC: closest approach → collision course
`t_cpa = −(dp·dv)/‖dv‖²` is the time of *closest approach*, not time to
*collision*. The engine now solves the actual conflict condition
`‖dp + dv·t‖ = R`, `R = r₁ + r₂ + buffer`:

```
‖dv‖² t² + 2(dp·dv) t + ‖dp‖² − R² = 0
```

The smallest non-negative root is the time until the footprints touch;
`t_cpa`/`d_cpa` remain as the fallback and the miss-distance gate. This makes
TTC consistent with the edge-to-edge distance philosophy the scoring already
used, and it is strictly earlier (more conservative) than `t_cpa` on true
collision courses.

### 3.2 Kalman filter parameters
White-acceleration CV model: `Q = σ_a²·[[dt⁴/4, dt³/2],[dt³/2, dt²]] ⊗ I₂`
with `σ_a = 2.5 m/s²` (urban traffic manoeuvre range), measurement noise
σ_m = 0.35 m (bbox jitter ≈ 2–4 px at typical ppm, plus homography error).
Initial velocity σ = max_speed/4 lets a fast vehicle converge within
~5 frames — inside the 15-frame `min_track_age_for_risk` warm-up.

### 3.3 Direction consistency
Kept the circular-resultant statistic (it is the right estimator:
`R̄ = ‖mean(cos θ, sin θ)‖`), but steps below 2 cm are excluded so a parked
vehicle's jitter cannot fake a consistent direction.

---

## 4. Collision detection (escalation beyond near-miss)

Footprint overlap alone is a false-positive machine: circular footprint
approximations of adjacent-lane traffic graze each other, and occluding
objects (one passing in front of another) can drive projected ground points
together with no crash. A confirmed collision therefore requires three
independent conditions (`_evaluate_collisions`, per-pair state machine):

1. **Contact** — edge distance ≤ `collision_overlap_m`, sustained for
   `collision_persist_frames`.
2. **Approach** — in the pre-contact window the pair was genuinely
   converging: max closing speed ≥ `collision_min_closing_mps` and max
   relative speed ≥ `collision_min_impact_rel_speed_mps`. Parallel traffic
   has closing ≈ 0 and is rejected here.
3. **Impact evidence** — within `collision_post_window_frames` of contact,
   momentum exchange must appear: a deceleration spike ≤
   `collision_impact_decel_mps2` on either track, relative speed collapsing
   below `(1 − drop_ratio) ×` its pre-impact value, or both tracks at rest.
   Occlusion pass-throughs separate at unchanged speed and never produce
   this, so the episode times out silently.

Confirmed collisions fire at a new risk level **Critical** (RI = 1.0,
magenta in overlays, own dashboard card), carry the reconstructed impact
relative speed and evidence list, and place the pair in a long cooldown that
also silences redundant near-miss alerts for the same pair. The near-miss
trail *before* the collision is intentionally preserved — it documents the
escalation.

Known trade-off: very low-speed taps (below
`collision_min_impact_rel_speed_mps`) are ignored by design — at CCTV
tracking noise levels they are indistinguishable from queueing traffic.
Vehicle-vs-infrastructure crashes (poles, barriers) are out of scope while
the detector only reports road users.

## 5. Remaining limitations (known, deliberate)

- **O(n²) pair loop** — fine to ~150 concurrent tracks; add a BEV grid hash
  if deployments exceed that.
- **Class table must match the model.** `class_mapping` is COCO-indexed; the
  custom `yolov8s_merger8` model uses a different order. Wire the mapping to
  the model's own `names` when switching.
- **Homography assumes a flat ground plane** — kerbs, ramps and bridges bias
  positions; calibrate per-camera and keep points inside the drivable area.
- **Single-point footprint** — orientation of the vehicle rectangle is not
  estimated; the circular-radius approximation over-buffers long vehicles
  seen side-on. Oriented boxes from heading are the natural next step.
- **`filter_reasonable_boxes` drops very large boxes** (> 35 % of frame) —
  intentional near-camera guard, but it also drops legitimate close-range
  buses; tune `track_max_box_ratio` per camera.
