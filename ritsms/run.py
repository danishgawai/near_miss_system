#!/usr/bin/env python3
"""RITSMS Near-Miss 2.0 — pipeline entry point (proposal-faithful).

ingest -> detect -> track -> trajectory(+forecast) -> conflict -> outputs

    python -m ritsms.run [--source X] [--max-frames N] [--no-video]
"""

import os
import sys
import csv
import time
import argparse
import logging
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ritsms.config import Config
from ritsms.ingest import FrameSource
from ritsms.detect import Detector
from ritsms.track import Tracker, TrackQualityMonitor
from ritsms.trajectory import TrackTrajectory
from ritsms.conflict import ConflictEngine
from ritsms.outputs import TrajectoryWriter, ConflictWriter, Reporter
from ritsms.calibration_check import check_config_match, CalibrationHealth, log_report
from utils.bev import BEVProjector
from utils.site import SiteConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ritsms")

_LEVEL_COLOR = {"CRITICAL": (0, 0, 255), "WARNING": (0, 165, 255)}


def _footprint_width_m(projector, bbox):
    x1, _, x2, y2 = bbox
    return projector.segment_length_m((float(x1), float(y2)), (float(x2), float(y2)))


def _away_dir(projector, gx, gy, step_px=12.0):
    """Ground-plane unit vector pointing away from the camera at (gx, gy).

    Obtained by projecting the reference point and a point slightly higher in the
    image (deeper into the scene) and taking the BEV difference. The vehicle body
    extends in this direction from its near-face ground contact, so it is what
    anchors the footprint correctly.
    """
    a = projector.to_bev_meters(gx, gy)
    b = projector.to_bev_meters(gx, max(0.0, gy - step_px))
    if a is None or b is None:
        return None
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n = float(np.linalg.norm(d))
    return (d / n) if n > 1e-6 else None


def _draw(frame, tracks, level_of, active, cfg, frame_idx, total, fps, site=None, projector=None):
    # ROI boundary (yellow) so the reference video shows what is analysed.
    if site is not None and site.roi_enabled and site.roi_polygon_img is not None:
        poly = site.roi_polygon_img.astype(int)
        cv2.polylines(frame, [poly], True, (0, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, "ROI", (int(poly[0][0][0]), max(14, int(poly[0][0][1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

    # level_of is the engine's LIVE per-track status this frame — a box stays
    # highlighted for the whole duration its pair is WARNING/CRITICAL, and
    # reverts to grey only once the pair is safe again.
    for tid, trk in tracks.items():
        if not trk.last_bbox:
            continue
        x1, y1, x2, y2 = trk.last_bbox
        lvl = level_of.get(tid)
        color = _LEVEL_COLOR.get(lvl, (150, 150, 150))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if lvl else 1)
        label = f"{tid} {trk.cls_name}" + (f" [{lvl}]" if lvl else "")
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Forecast path (predicted CV/CTRV trajectory) projected back to image.
        # Only after a track has accumulated a buffer of history (min_track_age)
        # so its velocity/heading/yaw — and thus the prediction — are stable.
        if (projector is not None and trk.age >= cfg.min_track_age and
                trk.speed >= cfg.min_speed_for_heading_mps):
            fc = trk.forecast(horizon_s=2.0, dt_s=0.25)
            pts = []
            for row in fc:
                ip = projector.to_image_px(row[1], row[2])
                if ip is None or not (0 <= ip[0] < frame.shape[1] and 0 <= ip[1] < frame.shape[0]):
                    break
                pts.append([int(ip[0]), int(ip[1])])
            if len(pts) >= 2:
                mode = trk.forecast_mode()
                fcol = (60, 220, 60) if mode == "CV" else (255, 130, 0)  # CTRV in blue-orange
                cv2.polylines(frame, [np.array(pts)], False, fcol, 2, cv2.LINE_AA)
                cv2.circle(frame, tuple(pts[-1]), 4, fcol, -1)
    y = 28
    for e in active[:6]:
        txt = f"[{e['level']}] {e['encounter_type']} {e['roaduser1_id']}<>{e['roaduser2_id']}"
        if e["ttc_s"] is not None:
            txt += f" TTC {e['ttc_s']}s"
        if e["pet_s"] is not None:
            txt += f" PET {e['pet_s']}s"
        cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    _LEVEL_COLOR.get(e["level"], (255, 255, 255)), 2, cv2.LINE_AA)
        y += 24
    cv2.putText(frame, f"F{frame_idx}/{total} @ {fps:.1f}fps", (12, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def run(cfg: Config, max_frames=None, verify_only=False):
    os.makedirs(cfg.output_dir, exist_ok=True)
    src = FrameSource(cfg.source, cfg.fps_ceiling)
    cfg.frame_width, cfg.frame_height = src.width, src.height
    fps = src.fps_eff

    # ---- Guard 1: does the site config actually belong to this video? ----
    problems = check_config_match(cfg.site_config_path, cfg.source, src.width, src.height)
    if problems:
        for p in problems:
            logger.error("CALIBRATION MISMATCH: %s", p)
        logger.error("Recalibrate for this camera:  python bev_web_calibrator.py "
                     "--video %s --config bev_config_<SITE>.json", cfg.source)
        if cfg.strict_calibration:
            src.release()
            raise SystemExit(
                "Aborting: the site config does not match the video being processed. "
                "Conflict measures would be geometrically invalid. Recalibrate, or pass "
                "--allow-config-mismatch to override deliberately."
            )
        logger.warning("Continuing despite mismatch (strict_calibration disabled) — "
                       "treat all metric output as invalid.")

    detector = Detector(cfg)
    logger.info("Warming detector (%d frames)...", cfg.warmup_frames)
    detector.warmup(cfg.warmup_frames)

    tracker = Tracker(cfg, fps)
    projector = BEVProjector(cfg.site_config_path, cfg.default_pixels_per_meter, cfg.bev_max_range_m)
    site = SiteConfig(cfg.site_config_path, "bev_only")
    engine = ConflictEngine(cfg, fps, site)
    qmon = TrackQualityMonitor(cfg)
    reporter = Reporter(cfg, site, fps)
    health = CalibrationHealth(
        stopped_speed_mps=cfg.calib_stopped_speed_mps,
        wander_warn_m=cfg.calib_wander_warn_m,
        wander_fail_m=cfg.calib_wander_fail_m,
        min_scene_span_m=cfg.calib_min_scene_span_m,
        max_scene_span_m=cfg.calib_max_scene_span_m,
        max_speed_mps=cfg.max_speed_mps,
    )

    # In verify mode nothing is written: this is a calibration check, not a run.
    if verify_only:
        cfg.write_video = False
        cfg.write_traces = False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_w = TrajectoryWriter(os.path.join(cfg.output_dir, f"trajectory_{ts}.csv"), site.metric_confident)
    conf_w = ConflictWriter(os.path.join(cfg.output_dir, f"conflicts_{ts}.csv"), site.metric_confident)
    writer = None
    if cfg.write_video:
        out_path = os.path.join(cfg.output_dir, f"annotated_{ts}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (src.width, src.height))
    trace_f = trace_wr = None
    if cfg.write_traces:
        trace_f = open(os.path.join(cfg.output_dir, f"traces_{ts}.csv"), "w", newline="", encoding="utf-8")
        trace_wr = csv.writer(trace_f)
        trace_wr.writerow(["frame", "t_s", "id1", "id2", "ttc", "pet", "level", "encounter"])

    tracks: dict = {}
    missing: dict = {}
    total = src.total // src.stride if src.total else 0
    start = time.time()
    n_conf = 0

    try:
        for pkt in src.frames():
            fi = pkt.index
            if max_frames and fi > max_frames:
                break

            boxes = detector.infer(pkt.frame)
            if site.roi_enabled:
                boxes = [b for b in boxes if site.point_in_image_roi(0.5 * (b[0] + b[2]), b[3])]
            live = tracker.update(boxes, src.width, src.height)

            active = set()
            for tid, box, score, cls_id in live:
                active.add(tid)
                missing[tid] = 0
                x1, y1, x2, y2 = np.asarray(box).astype(int)
                gx, gy = 0.5 * (x1 + x2), float(y2)
                cls = cfg.class_mapping.get(int(cls_id), "vehicle")
                if tid not in tracks:
                    tracks[tid] = TrackTrajectory(tid, cls, cfg)
                trk = tracks[tid]
                dt = max(1, fi - trk.last_update_frame if trk.last_update_frame > 0 else 1) / fps
                bev = projector.to_bev_meters(gx, gy)
                trk.update(bev, [int(x1), int(y1), int(x2), int(y2)], (int(gx), int(gy)),
                           score, fi, dt, _footprint_width_m(projector, box),
                           away_dir=_away_dir(projector, gx, gy))

            for tid in list(tracks):
                if tid not in active:
                    missing[tid] = missing.get(tid, 0) + 1
                    if missing[tid] > cfg.track_drop_after_missed:
                        tracks.pop(tid, None)
                        missing.pop(tid, None)

            positions = {tid: t.position_m for tid, t in tracks.items() if t.ready}
            jumps = sum(t.jump_count for t in tracks.values())
            qmon.update(fi, positions, jump_delta=0)
            health.observe(tracks)

            incidents = engine.evaluate(fi, pkt.capture_ts_ms, None, tracks, pkt.t_s)
            n_conf += len(incidents)
            traj_w.write_frame(fi, pkt.capture_ts_ms, None, tracks)
            for e in incidents:
                conf_w.write(e)
            if trace_wr is not None:
                for tr in engine.frame_traces:
                    trace_wr.writerow([tr["frame"], tr["t_s"], tr["id1"], tr["id2"],
                                       tr["ttc"], tr["pet"], tr["level"], tr["encounter"]])
            reporter.add(incidents)

            if writer is not None:
                writer.write(_draw(pkt.frame, tracks, engine.frame_levels, engine.frame_active,
                                   cfg, fi, total, fps, site, projector))

            if fi % 50 == 0 or incidents:
                logger.info("F%d/%s det=%d trk=%d conf+=%d total=%d",
                            fi, total or "?", len(boxes), len(live), len(incidents), n_conf)
            if time.time() - start > cfg.run_time_seconds:
                logger.info("Runtime limit reached."); break
    finally:
        src.release()
        if writer is not None:
            writer.release()
        if trace_f is not None:
            trace_f.close()
        traj_w.close()
        conf_w.close()

    # ---- Guard 2: empirical calibration health from what we just processed ----
    hrep = health.report()
    log_report(hrep)

    if verify_only:
        # Clean up the empty artefacts a verify pass shouldn't leave behind.
        for p in (os.path.join(cfg.output_dir, f"trajectory_{ts}.csv"),
                  os.path.join(cfg.output_dir, f"conflicts_{ts}.csv")):
            try:
                os.remove(p)
            except OSError:
                pass
        return hrep

    tq = qmon.summary()
    tq["jump_jitter_events"] = sum(t.jump_count for t in tracks.values())  # residual live tracks
    s = reporter.save(
        os.path.join(cfg.output_dir, f"report_{ts}.json"),
        os.path.join(cfg.output_dir, f"dashboard_{ts}.html"),
        os.path.join(cfg.output_dir, f"heatmap_{ts}.png"),
        tq,
    )
    logger.info("DONE. conflicts=%d  outputs in %s/", s["total_conflicts"], cfg.output_dir)
    logger.info("levels=%s governing=%s", s["level_distribution"], s["governing_measure"])
    if hrep["overall"] == "FAIL":
        logger.error("NOTE: calibration health FAILED for this run — the conflict numbers "
                     "above are not trustworthy. See the calibration health block.")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--device", default=None, help='"cpu" | "cuda" | "0" (GPU needs a CUDA torch)')
    ap.add_argument("--model", default=None, help="detector path; use the .pt for GPU")
    ap.add_argument("--half", action="store_true", help="FP16 (Pascal+ GPUs only)")
    ap.add_argument("--site-config", default=None,
                    help="per-site bev config (homography + ROI), e.g. bev_config_IP33B.json")
    ap.add_argument("--site-id", default=None, help="site identifier written to the outputs")
    ap.add_argument("--verify-calibration", action="store_true",
                    help="run a short segment and only report calibration health; writes nothing")
    ap.add_argument("--verify-frames", type=int, default=600,
                    help="frames to use for --verify-calibration (default 600)")
    ap.add_argument("--allow-config-mismatch", action="store_true",
                    help="proceed even if the site config doesn't match the source (unsafe)")
    args = ap.parse_args()
    cfg = Config()
    if args.source:
        cfg.source = args.source
    if args.no_video:
        cfg.write_video = False
    if args.device:
        cfg.model_device = args.device
    if args.model:
        cfg.model_path = args.model
    if args.half:
        cfg.model_half = True
    if args.site_config:
        cfg.site_config_path = args.site_config
    if args.site_id:
        cfg.site_id = args.site_id
    if args.allow_config_mismatch:
        cfg.strict_calibration = False

    if args.verify_calibration:
        # A verification pass must not be blocked by the static guard — reporting
        # the empirical health is exactly the point.
        cfg.strict_calibration = False
        rep = run(cfg, max_frames=args.verify_frames, verify_only=True)
        raise SystemExit(0 if rep["overall"] in ("PASS", "WARN") else 1)

    run(cfg, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
