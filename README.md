# Near-Miss Detection (Quick Test Guide)

This project provides:

1. **Simple detector + tracker test** (`run_tracker.py`)  
2. **BEV calibration tool** (`bev_calibrator.py`)  
3. **Main near-miss pipeline** (`main.py`) using BEV + TTC + risk scoring  

---

## 1) Setup

### Requirements
- Python 3.9+ (recommended)
- CPU environment (project is CPU-friendly)
- OpenCV GUI support (for calibration windows)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2) Quick detector + tracker test (`run_tracker.py`)

Use this first to verify model loading, detection, and tracking.

### Edit test source
In `run_tracker.py`, set:

- `source_stream = "your_video.mp4"`

### Run
```bash
python run_tracker.py
```

### Expected output
- Console logs for frame processing and FPS
- Output video file like: `video_out_YYYY-MM-DD_HH:MM:SS.mp4`
- Track trails drawn on moving objects

> Note: This script is only for quick testing (no BEV / near-miss logic).

---

## 3) BEV calibration (`bev_calibrator.py`)

Run calibration before the main near-miss app.

### Run
```bash
python bev_calibrator.py --video your_video.mp4 --frame-index 100 --output-config bev_config.json
```

### Calibration workflow
A window opens with:
- **Left**: real frame (source points)
- **Right**: BEV canvas (destination points)

#### Controls
- Left click: select points (alternate left/right)
- `r`: reset points
- `Enter`: compute homography (need >= 4 matched pairs)
- In BEV preview:
  - `s`: continue to scale calibration and save
  - `q`: quit without saving

#### Best practice
- Use **8–12 point pairs** spread across the road area
- Avoid collinear points
- Prefer corners/line intersections/markings visible in both views

### Scale calibration
After homography preview:
1. Click 2 points on BEV image with known real distance
2. Press `Enter`
3. Input distance in meters (example: lane width)

This computes `pixels_per_meter`.

### Output
- `bev_config.json` containing:
  - homography matrix
  - source/destination points
  - pixels_per_meter

---

## 4) Run main near-miss pipeline (`main.py`)

Ensure config paths are correct in `config.py`:
- `source_stream`
- `bev_config_path` (usually `bev_config.json`)
- model path

### Run
```bash
python main.py
```

### Outputs
- Annotated output video
- `near_miss_report.json`
- `near_miss_dashboard.html`
- High-risk frame snapshots (if enabled)

---

## 5) Minimal test order (recommended)

1. `python run_tracker.py` → confirm detector/tracker works  
2. `python bev_calibrator.py ...` → generate valid `bev_config.json`  
3. `python main.py` → run near-miss analysis  

---

## 6) Common issues

### No calibration window appears
- Use local desktop Python (not headless server)
- Ensure OpenCV GUI backend is available

### BEV distances look wrong
- Recalibrate with better spread points
- Recheck scale distance input
- Ensure same camera/video as runtime

### Too many false near-miss alerts
- Increase `incident_persist_frames`
- Increase `min_closing_speed_mps`
- Reduce `proximity_gate_m`
- Increase detector confidence

---

## 7) Example commands

```bash
# 1) Tracker smoke test
python run_tracker.py

# 2) BEV calibration
python bev_calibrator.py --video traffic_clipped01.mp4 --frame-index 120 --output-config bev_config.json

# 3) Main near-miss pipeline
python main.py
```

---

## Notes
- `run_tracker.py` is intended for **testing only**.
- For project/demo outputs, use calibrated BEV + `main.py`.