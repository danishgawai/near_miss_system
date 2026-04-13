# Near-Miss Incident Detection System (CPU-Friendly)

A practical computer-vision pipeline for traffic safety analysis:

- **Object Detection + Tracking**
- **BEV (Bird’s-Eye View) projection**
- **Near-miss logic (proximity + trajectory + TTC + hard braking)**
- **Risk scoring (High / Medium / Low)**
- **Video annotations + JSON report + HTML dashboard**

This repository also includes:
- `run_tracker.py` for quick detector/tracker smoke testing
- `bev_calibrator.py` for homography + scale calibration

---

## Project Components

### 1) Quick Test Script
- **File**: `run_tracker.py`
- Purpose: verify detector and tracker are working correctly on a sample video.
- Output: annotated tracking video.

### 2) BEV Calibration Tool
- **File**: `bev_calibrator.py`
- Purpose: create `bev_config.json` with:
  - Homography matrix
  - Calibration points
  - Pixels-per-meter scale

### 3) Main Near-Miss Application
- **File**: `main.py`
- Purpose: full pipeline for near-miss detection and reporting.

---

## Suggested Project Structure

```text
project_root/
├─ main.py
├─ config.py
├─ run_tracker.py
├─ bev_calibrator.py
├─ requirements.txt
├─ README.md
├─ models/
│  └─ yolo26n_int8_openvino_model/
├─ utils/
│  ├─ yolo_infer.py
│  ├─ near_miss.py
│  ├─ tracker_manager.py
│  ├─ motion.py
│  ├─ visualization.py
│  ├─ reporting.py
│  └─ bev.py
└─ tracker/
   ├─byte_tracker.py
   ├─basetrack.py
   └─ utils/
```

---

## Requirements

- Python 3.9+ recommended
- CPU environment (optimized for CPU usage)
- OpenCV GUI support for calibration windows

Create a virtual Environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
# Note: If running on CPU, first install torch for CPU.
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
ultralytics 
openpyxl 
pandas 
opencv-python 
lap 
cython_bbox 
shapely
```

---

## Quick Start (Recommended Order)

1. **Run tracker smoke test**
2. **Calibrate BEV**
3. **Run main near-miss pipeline**

---

## 1) Run Simple Detector + Tracker (Testing)

Use this to quickly validate model/tracker behavior before full pipeline.

### Edit source video
In `run_tracker.py`:
```python
source_stream = "your_video.mp4"
```

### Run
```bash
python run_tracker.py
```

### Expected
- Console FPS/frame logs
- Output annotated video (e.g., `video_out_*.mp4`)
- Track trails visible on moving objects

> Note: This script is only for testing and does not perform near-miss analytics.

---

## 2) BEV Calibration

Before running near-miss logic, generate a reliable BEV config.

### Run
```bash
python bev_calibrator.py --video your_video.mp4 --frame-index 100 --output-config bev_config.json
```

### Calibration UI
- **Left panel**: real camera frame (source points)
- **Right panel**: BEV canvas (destination points)

### Controls
- Left click: add points (alternate left/right)
- `r`: reset
- `Enter`: compute homography (requires >= 4 matched pairs)
- Preview window:
  - `s`: save and continue to scale calibration
  - `q`: quit without saving

### Best Practices
- Use **4–8 matched pairs** 
- Spread points across full drivable area
- Avoid collinear points
- Use stable landmarks (lane corners, stop-line intersections, road markings)

### Scale Calibration
After homography preview:
1. Click 2 points with known real-world distance
2. Press `Enter`
3. Enter distance in meters (e.g., lane width)

Tool computes `pixels_per_meter` and saves config.

### Output
`bev_config.json` with homography + scale parameters.

---

## 3) Run Main Near-Miss Application

Ensure `config.py` points to:
- video source
- model path
- `bev_config.json`

### Run
```bash
python main.py
```

### Main Outputs
- Annotated output video (`video_out_*.mp4`)
- `near_miss_report.json`
- `near_miss_dashboard.html`
- High-risk frame snapshots (if enabled)

---

## Near-Miss Logic Summary

The main app evaluates near-miss events using:
- Proximity gate (distance threshold in BEV meters)
- Relative motion and trajectory conflict
- TTC (time-to-collision) estimation
- Hard-braking events
- Risk matrix (probability × severity)

Events are tagged with:
- timestamp
- frame index
- actor IDs/classes
- TTC/distance/speed metrics
- risk level (High/Medium/Low)

---

## Basic Tuning Guide

If too many false alerts:
- Increase `incident_persist_frames`
- Increase `min_closing_speed_mps`
- Reduce `proximity_gate_m`
- Increase `model_conf`

If missing incidents:
- Reduce `model_conf`
- Increase `proximity_gate_m`
- Reduce `incident_persist_frames`
- Verify BEV calibration quality

If hard-braking is too noisy:
- decrease sensitivity by lowering threshold (e.g., `-4.5` → `-5.5`)
- increase `hard_brake_min_speed_mps`

---

## Common Issues

### 1) Missing Detections and tracks
- Current Coco model miss detections when vehicles moves further from camera.
- Due to missed detections, frequent track id switch is observed, reducing overall event accuracy.

### 2) Distances/TTC unrealistic
- Bad homography point selection
- Wrong scale distance input
- Using different camera/video than calibrated source

### 3) Tracker ID switching
- Tune `match_thresh`, `track_buffer`
- Increase tracking stabilization
- Centroid Point selection not optimal for accurate events.
- Consider stronger tracker variant (future scope)

---

## Example Commands

```bash
# 1) Quick detector+tracker test
python run_tracker.py

# 2) BEV calibration
python bev_calibrator.py --video traffic_clipped01.mp4 --frame-index 120 --output-config bev_config.json

# 3) Main near-miss pipeline
python main.py
```

---

## Future Scope / Improvements

### Detection & Tracking
- Fine-tune model for consistent detections and tracking.
- Add re-identification embeddings to reduce ID switches after occlusion.

### Motion & Forecasting
- Replace simple TTC with multi-step trajectory forecasting (Kalman/CA models).
- Add curvature/lane-aware trajectory constraints for intersections.

### Optical Flow & Scene Context
- Improve optical-flow integration for occlusion-heavy scenes.
- Add lane/road segmentation for context-aware filtering.

### False Positive Reduction
- Adaptive thresholds by traffic density and object class pair.
- Event verification stage for uncertain incidents.
- Raise events for individual ID's or group of vehicle involved in the incident.

### Severity Scoring
- Train a lightweight ML severity model using:
  TTC trend, distance trend, heading conflict, speed delta, braking profile.
- Keep explainable outputs for reporting.

### Productization
- Real-Time processing (RTSP in, alert out).
- REST API for incidents and analytics.
- Dockerized deployment with CPU/GPU profiles.

---

## Author

- Author: Danish Ahmed
- Mail: Danishh163@gmail.com
- Status: Development 
