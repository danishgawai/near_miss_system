#!/usr/bin/env python3
"""Near-Miss Incident Detection System — main pipeline entry point."""

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
from utils.site import SiteConfig
from utils.motion import MotionEstimator, TrackState
from utils.track_manager import TrackerManager
from utils.near_miss import NearMissEngine
from utils.visualization import draw_frame
from utils.reporting import Reporter, TelemetryLogger, save_dashboard
from utils.yolo_infer import YOLO_Inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _footprint_radius_m(cfg, projector, cls_name, x1, y2_, x2) -> float:
    """Effective circular footprint radius in metres.

    Half the bounding box's bottom edge, projected through the homography
    (a ground-plane segment, so metric length is well-defined), clipped to
    a plausible band around the class prior. Falls back to the prior when
    the segment is unprojectable.
    """
    prior = cfg.class_radius_m.get(cls_name, 1.4)
    width_m = projector.segment_length_m((float(x1), float(y2_)), (float(x2), float(y2_)))
    if width_m is None or width_m <= 0:
        return prior
    return float(np.clip(0.5 * width_m, 0.5 * prior, 2.0 * prior))


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
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or not (1.0 <= fps <= 120.0):
        logging.warning(f"Unreliable source FPS ({fps}); falling back to 25.0. "
                        "All speeds/TTC scale with FPS — verify this value.")
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pass actual frame dimensions to config for the border guard
    cfg.frame_width = w
    cfg.frame_height = h

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video = cfg.output_video_path or f"video_out_{ts}.mp4"
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {out_video}")

    tracker = TrackerManager(cfg.track_thresh, cfg.track_buffer, cfg.match_thresh, fps)
    projector = BEVProjector(cfg.bev_config_path, cfg.default_pixels_per_meter,
                             cfg.bev_max_range_m)
    site = SiteConfig(cfg.site_config_path, cfg.calibration_confidence_default)
    motion = MotionEstimator(
        fps,
        process_noise_accel=cfg.kf_process_noise_accel,
        measurement_noise_m=cfg.kf_measurement_noise_m,
        gate_chi2=cfg.kf_gate_chi2,
        outlier_reset_frames=cfg.kf_outlier_reset_frames,
        max_speed_mps=cfg.max_speed_mps,
    )
    risk_engine = NearMissEngine(cfg, fps, site)
    reporter = Reporter(fps, site.calibration_confidence)
    telemetry = TelemetryLogger(cfg.telemetry_csv_path)
    telemetry.open()

    tracks_state: dict = {}
    track_history = defaultdict(lambda: deque(maxlen=cfg.history_len))
    missing_count = defaultdict(int)

    frame_idx = 0
    start = time.time()
    total_time = 0.0
    total_incidents = 0

    try:
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

            # 1b) ROI gate — when an ROI is configured, only detections whose
            # ground-contact point (bbox bottom-centre) falls inside it are
            # processed; "full frame" mode leaves this a no-op.
            if site.roi_enabled:
                boxes = [b for b in boxes
                         if site.point_in_image_roi(0.5 * (b[0] + b[2]), b[3])]

            # 2) Track
            raw = tracker.update(boxes)
            valid_tracks = tracker.filter_reasonable_boxes(raw, w, h, cfg.track_max_box_ratio)

            # 3) Update track states
            active_ids = set()
            for tid_raw, centroid, box, score, cls_id in valid_tracks:
                tid = int(tid_raw)
                active_ids.add(tid)
                missing_count[tid] = 0

                x1, y1, x2, y2 = np.asarray(box).astype(np.int32)
                # Ground-contact point: bottom-centre of the bbox. The
                # homography maps the GROUND plane, so only points on it
                # project correctly — the bbox centre sits ~half the object
                # height above ground and lands metres away in BEV.
                gx = 0.5 * (x1 + x2)
                gy = float(y2)
                cls_name = cfg.class_mapping.get(int(cls_id), "vehicle")

                if tid not in tracks_state:
                    tracks_state[tid] = TrackState(track_id=tid, cls_name=cls_name)

                trk = tracks_state[tid]
                trk.cls_name = cls_name
                trk.det_score = float(score)
                trk.last_bbox = [int(x1), int(y1), int(x2), int(y2)]
                trk.img_points.append((int(gx), int(gy)))
                trk.bev_radius_m = _footprint_radius_m(cfg, projector, cls_name, x1, y2, x2)

                bev_pos = projector.to_bev_meters(gx, gy)  # None if unprojectable
                motion.update_track(trk, bev_pos, frame_idx)

                track_history[tid].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

            # Handle temporarily missing tracks
            for tid in list(tracks_state.keys()):
                if tid not in active_ids:
                    missing_count[tid] += 1
                    if missing_count[tid] > cfg.track_drop_after_missed:
                        del tracks_state[tid]
                        missing_count.pop(tid, None)
                        track_history.pop(tid, None)

            risk_engine.prune(tracks_state.keys())

            # 4) Risk evaluation
            now = datetime.now().isoformat()
            video_time_s = frame_idx / fps
            incidents = risk_engine.evaluate(frame_idx, now, tracks_state, video_time_s)
            reporter.add(incidents)
            total_incidents += len(incidents)

            # 4b) Telemetry
            ri_by_id = {}
            for inc in incidents:
                ri = inc.get("risk_index", 0.0) or 0.0
                for k in ("actor_1_id", "actor_2_id"):
                    tid = inc.get(k)
                    if tid is not None:
                        ri_by_id[tid] = max(ri_by_id.get(tid, 0.0), ri)
            for tid in active_ids:
                trk = tracks_state.get(tid)
                if trk is None or not trk.bev_positions:
                    continue
                pos = trk.bev_positions[-1]
                telemetry.log(frame_idx, now, tid, trk.cls_name,
                              pos[0], pos[1], trk.speed, trk.heading_deg,
                              trk.acc, ri_by_id.get(tid, 0.0))

            # 5) Draw + write
            loop = time.time() - t0
            total_time += loop
            cur_fps = 1.0 / loop if loop > 0 else 0.0
            avg_fps = frame_idx / total_time if total_time > 0 else 0.0

            annotated = draw_frame(
                annotated, valid_tracks, cfg.class_mapping, track_history, incidents,
                frame_idx, total_frames, cur_fps, avg_fps, total_incidents,
            )
            writer.write(annotated)

            collisions = [x for x in incidents if x.get("type") == "Collision"]
            critical = [x for x in incidents if x.get("level") == "CRITICAL"]
            warning = [x for x in incidents if x.get("level") == "WARNING"]
            if critical or warning:
                os.makedirs(cfg.high_risk_frame_dir, exist_ok=True)
                snap = os.path.join(cfg.high_risk_frame_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(snap, annotated)
                if collisions:
                    for c in collisions:
                        logging.critical(
                            f"COLLISION frame {frame_idx}: {c['actor_1']} vs {c['actor_2']} "
                            f"impact={c['impact_rel_speed_kmh']} km/h "
                            f"evidence={c['evidence']} → {snap}"
                        )
                elif critical:
                    logging.critical(
                        f"CRITICAL frame {frame_idx}: {len(critical)} conflict(s) → {snap}"
                    )
                else:
                    logging.warning(
                        f"WARNING frame {frame_idx}: {len(warning)} conflict(s) → {snap}"
                    )

            if frame_idx % 30 == 0 or incidents:
                logging.info(
                    f"F{frame_idx}/{total_frames}  det={len(boxes)}  "
                    f"trk={len(valid_tracks)}  inc={len(incidents)}  "
                    f"fps={cur_fps:.1f}  avg={avg_fps:.1f}"
                )

            if time.time() - start > cfg.run_time_seconds:
                logging.info("Runtime limit reached.")
                break
    finally:
        cap.release()
        writer.release()
        telemetry.close()

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
