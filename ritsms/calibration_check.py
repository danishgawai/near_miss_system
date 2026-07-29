"""Calibration guards.

Catches the failure mode where a site's homography/ROI is applied to a DIFFERENT
camera's video: BEV positions come out geometrically wrong while still looking
plausible, so nothing complains.

  * check_config_match()  - static and RELIABLE. Does the site config actually
    belong to the video about to be processed (source basename, frame
    dimensions, homography present)? This is the check that catches the mistake
    in practice, and it is the one wired to abort the run.

  * CalibrationHealth     - COARSE SANITY ONLY. Measured on a short segment:
      - scene span         (FAIL: geometry collapsed or exploding)
      - speed plausibility (FAIL: impossible speeds)
      - stopped-track BEV wander (INFO only)

    Scope limits, measured rather than assumed: stopped-track wander,
    depth-vs-width correlation and depth-vs-speed correlation were all tested
    against a known-good and a known-bad calibration and NONE of them separated
    the two (wander 0.50 m vs 0.51 m; corr(depth,width) +0.46 vs +0.56). The
    reason is that an unvalidated 4-point `bev_only` homography is itself
    inaccurate, so there is no trustworthy reference to compare against. These
    checks therefore only catch catastrophically broken geometry; they are NOT
    a substitute for calibration validation.

    Rigorous validation requires metric ground-control points and reprojection
    error (RMSE / P95) per proposal AC1 - i.e. leaving `bev_only` mode.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ── static config-vs-source match ─────────────────────────────────────────

def check_config_match(site_cfg_path, source, frame_w, frame_h) -> list:
    """Return a list of human-readable problems (empty == consistent)."""
    import json
    problems = []
    if not os.path.isfile(site_cfg_path):
        return [f"site config '{site_cfg_path}' does not exist"]
    try:
        with open(site_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        return [f"site config '{site_cfg_path}' unreadable: {e}"]

    cfg_src = cfg.get("video_source")
    if cfg_src:
        # Compare basenames: paths legitimately differ between calibrate/run.
        if os.path.basename(str(cfg_src)) != os.path.basename(str(source)):
            problems.append(
                f"calibrated for '{os.path.basename(str(cfg_src))}' but processing "
                f"'{os.path.basename(str(source))}' - the homography/ROI belong to a "
                f"different camera view"
            )
    cw, ch = cfg.get("frame_width"), cfg.get("frame_height")
    if cw and ch and (int(cw) != int(frame_w) or int(ch) != int(frame_h)):
        problems.append(
            f"calibrated at {cw}x{ch} but the stream is {frame_w}x{frame_h} - "
            f"image coordinates (and the ROI) will not line up"
        )
    if not cfg.get("homography_matrix"):
        problems.append("no homography_matrix in the site config (BEV output will be garbage)")
    return problems


# ── empirical health from a processed segment ──────────────────────────────

class CalibrationHealth:
    """Accumulates per-track BEV samples, then judges calibration sanity.

    Feed it with observe() each processed frame; call report() at the end.
    """

    def __init__(self, stopped_speed_mps=0.5, wander_warn_m=2.0, wander_fail_m=4.0,
                 min_scene_span_m=20.0, max_scene_span_m=400.0, max_speed_mps=40.0):
        self.stopped_speed = float(stopped_speed_mps)
        self.wander_warn = float(wander_warn_m)
        self.wander_fail = float(wander_fail_m)
        self.min_span = float(min_scene_span_m)
        self.max_span = float(max_scene_span_m)
        self.max_speed = float(max_speed_mps)
        self._pos = {}      # tid -> list of (x, y)
        self._spd = {}      # tid -> list of speed
        self.frames = 0

    def observe(self, tracks: dict):
        self.frames += 1
        for tid, trk in tracks.items():
            if not getattr(trk, "ready", False):
                continue
            p = trk.position_m
            self._pos.setdefault(tid, []).append((float(p[0]), float(p[1])))
            self._spd.setdefault(tid, []).append(float(trk.speed))

    # -- metrics --
    def _wanders(self, min_run=8):
        """BEV spread measured over CONTIGUOUS runs of stopped samples.

        Measuring a whole track's extent is wrong: a vehicle decelerating into a
        queue has a low median speed yet legitimately covers its approach
        distance. Only frames where the track is actually stopped count, and only
        in contiguous runs, so a track that stops at two different places does
        not register the travel between them.
        """
        out = []
        for tid, pts in self._pos.items():
            spd = self._spd.get(tid) or []
            if len(pts) < min_run or len(spd) != len(pts):
                continue
            run = []
            for p, s in list(zip(pts, spd)) + [(None, None)]:   # sentinel flushes
                if s is not None and s < self.stopped_speed:
                    run.append(p)
                    continue
                if len(run) >= min_run:
                    a = np.asarray(run, dtype=np.float64)
                    span = a.max(axis=0) - a.min(axis=0)
                    out.append(float(np.hypot(span[0], span[1])))
                run = []
        return out

    def _scene_span(self):
        if not self._pos:
            return 0.0, 0.0
        a = np.asarray([p for pts in self._pos.values() for p in pts], dtype=np.float64)
        return float(a[:, 0].max() - a[:, 0].min()), float(a[:, 1].max() - a[:, 1].min())

    def report(self) -> dict:
        wanders = self._wanders()
        sx, sy = self._scene_span()
        speeds = [s for v in self._spd.values() for s in v]
        checks = []

        # 1) stopped-track wander - INFO ONLY. Measured on known-good vs
        # known-bad calibrations it did not separate them (0.50 m vs 0.51 m), so
        # it must not drive a verdict. Reported because a very large value is
        # still worth a human look.
        if len(wanders) >= 3:
            med = float(np.median(wanders))
            mx = float(np.max(wanders))
            checks.append({
                "check": "stopped_track_bev_wander", "status": "INFO",
                "detail": f"median {med:.2f} m, max {mx:.2f} m over {len(wanders)} stopped segments "
                          f"(informational: does NOT validate calibration - see AC1/RMSE)",
                "median_m": round(med, 2), "max_m": round(mx, 2), "n": len(wanders),
            })
        else:
            checks.append({
                "check": "stopped_track_bev_wander", "status": "SKIP",
                "detail": "too few stopped segments observed (need >=3)",
            })

        # 2) scene span plausibility
        span = max(sx, sy)
        st = "PASS" if self.min_span <= span <= self.max_span else "FAIL"
        checks.append({
            "check": "scene_span", "status": st,
            "detail": f"BEV extent {sx:.1f} x {sy:.1f} m (expect roughly "
                      f"{self.min_span:.0f}-{self.max_span:.0f} m for a junction; too small "
                      f"means the scale is compressed, too large means it is exploding)",
            "span_x_m": round(sx, 1), "span_y_m": round(sy, 1),
        })

        # 3) speed plausibility
        if speeds:
            p95 = float(np.percentile(speeds, 95))
            mx = float(np.max(speeds))
            st = "PASS" if mx <= self.max_speed else "FAIL"
            checks.append({
                "check": "speed_plausibility", "status": st,
                "detail": f"p95 {p95:.1f} ({p95*3.6:.0f} km/h), max {mx:.1f} "
                          f"({mx*3.6:.0f} km/h); ceiling {self.max_speed:.0f} m/s",
                "p95_mps": round(p95, 2), "max_mps": round(mx, 2),
            })

        worst = "PASS"
        for c in checks:
            if c["status"] == "FAIL":
                worst = "FAIL"
                break
            if c["status"] == "WARN":
                worst = "WARN"
        return {"overall": worst, "frames": self.frames,
                "tracks_observed": len(self._pos), "checks": checks}


def log_report(rep: dict) -> None:
    icon = {"PASS": "OK  ", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "SKIP", "INFO": "INFO"}
    logger.info("Calibration sanity (coarse): %s (%d frames, %d tracks)",
                rep["overall"], rep["frames"], rep["tracks_observed"])
    for c in rep["checks"]:
        logger.info("  [%s] %s: %s", icon.get(c["status"], c["status"]), c["check"], c["detail"])
    if rep["overall"] == "FAIL":
        logger.error("BEV geometry is badly broken for this video - recalibrate with "
                     "bev_web_calibrator.py against THIS camera before running analysis.")
    else:
        logger.info("NOTE: this is a coarse smoke test, not calibration validation. "
                    "Rigorous checking needs metric ground-control points and "
                    "reprojection RMSE/P95 (proposal AC1).")
