#!/usr/bin/env python3
import os
import time
import logging
import traceback
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict, deque

from config import AppConfig
from utils.bev import BEVProjector
from utils.motion import MotionEstimator, TrackState
from utils.track_manager import TrackerManager
from utils.near_miss import NearMissEngine
from utils.visualization import draw_frame
from utils.reporting import Reporter, save_dashboard
from utils.yolo_infer import YOLO_Inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run():
    cfg = AppConfig()

    detector = YOLO_Inference(
        model_path=cfg.model_path,
        conf_thres=cfg.model_conf,
        device=cfg.model_device,
        filter_class_ids=cfg.filter_class_ids,
    )

    logging.info(f"Warming model for {cfg.warmup_frames} frames...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(cfg.warmup_frames):
        detector.infer(dummy)

    cap = cv2.VideoCapture(cfg.source_stream)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {cfg.source_stream}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video = cfg.output_video_path or f"video_out_{ts}.mp4"
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = TrackerManager(cfg.track_thresh, cfg.track_buffer, cfg.match_thresh, fps)
    projector = BEVProjector(cfg.bev_config_path, cfg.default_pixels_per_meter)
    motion = MotionEstimator(fps, cfg.velocity_alpha, cfg.accel_alpha)
    risk_engine = NearMissEngine(cfg)
    reporter = Reporter(fps)

    tracks_state = {}
    track_history = defaultdict(lambda: deque(maxlen=cfg.history_len))
    missing_count = defaultdict(int)

    frame_idx = 0
    start = time.time()
    total_time = 0.0
    total_incidents = 0
    max_missing_keep = 8  # keep short occlusions for stability

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        t0 = time.time()
        annotated = frame.copy()

        # 1) Detect
        try:
            boxes = detector.infer(frame)
        except Exception as e:
            logging.error(f"Detection error @ frame {frame_idx}: {e}")
            boxes = []

        # 2) Track
        raw = tracker.update(boxes)
        valid_tracks = tracker.filter_reasonable_boxes(raw, w, h, cfg.track_max_box_ratio)

        # 3) Update track states
        active_ids = set()
        for tid_raw, centroid, box, score, cls_id in valid_tracks:
            tid = int(tid_raw)
            active_ids.add(tid)
            missing_count[tid] = 0

            x1, y1, x2, y2 = box.astype(np.int32)
            cx = int((x1 + x2) / 2)
            cy = int(y2)  # contact-ish point on road
            cls_name = cfg.class_mapping.get(int(cls_id), "vehicle")

            if tid not in tracks_state:
                tracks_state[tid] = TrackState(track_id=tid, cls_name=cls_name)

            trk = tracks_state[tid]
            trk.cls_name = cls_name
            trk.last_bbox = [int(x1), int(y1), int(x2), int(y2)]
            trk.img_points.append((cx, cy))

            bev_pos = projector.to_bev_meters(float(cx), float(cy))
            motion.update_track(trk, bev_pos)

            track_history[tid].append((cx, cy))

        # handle temporarily missing tracks
        for tid in list(tracks_state.keys()):
            if tid not in active_ids:
                missing_count[tid] += 1
                if missing_count[tid] > max_missing_keep:
                    del tracks_state[tid]
                    missing_count.pop(tid, None)

        for tid in list(track_history.keys()):
            if tid not in tracks_state:
                del track_history[tid]

        # 4) Risk eval
        now = datetime.now().isoformat()
        incidents = risk_engine.evaluate(frame_idx, now, tracks_state)
        reporter.add(incidents)
        total_incidents += len(incidents)

        # 5) Draw + write
        loop = time.time() - t0
        total_time += loop
        cur_fps = 1.0 / loop if loop > 0 else 0.0
        avg_fps = frame_idx / total_time if total_time > 0 else 0.0

        annotated = draw_frame(
            annotated, valid_tracks, cfg.class_mapping, track_history, incidents,
            frame_idx, total_frames, cur_fps, avg_fps, total_incidents
        )
        writer.write(annotated)

        high = [x for x in incidents if x.get("risk") == "High"]
        if high:
            os.makedirs(cfg.high_risk_frame_dir, exist_ok=True)
            snap = os.path.join(cfg.high_risk_frame_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(snap, annotated)
            logging.warning(f"HIGH RISK frame {frame_idx}: {len(high)} incidents -> {snap}")

        if frame_idx % 30 == 0 or incidents:
            logging.info(
                f"F{frame_idx}/{total_frames} det={len(boxes)} trk={len(valid_tracks)} "
                f"inc={len(incidents)} fps={cur_fps:.1f} avg={avg_fps:.1f}"
            )

        if time.time() - start > cfg.run_time_seconds:
            logging.info("Runtime limit reached.")
            break

    cap.release()
    writer.release()
    logging.info(f"Output video saved: {out_video}")

    summary = reporter.save_json(cfg.report_output_path)
    save_dashboard(summary, cfg.dashboard_output_path)
    logging.info("Run complete.")


if __name__ == "__main__":
    print("=" * 64)
    print(" Near-Miss Incident Detection System")
    print("=" * 64)
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logging.error(f"Fatal error: {e}\n{traceback.format_exc()}")
    print("Done.")