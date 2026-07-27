"""§5 — Acceptance-evidence generator.

Post-processes a pipeline run's outputs (report_*.json + traces_*.csv +
trajectory_*.csv in the output dir) into the artifacts the TTS validation
protocol asks for:

  * AC4 — 15-minute traffic-volume counts by road-user class and movement
          type (through / left / right / u-turn), as a table + bar chart.
  * AC6 — TTC-vs-frame plots for sampled conflicts (with the 3 s / 1.5 s
          screening lines and the 0.5 s NFb3 window marked) for frame-by-frame
          review.
  * AC3 — trajectory-integrity summary (lifespans, speed distribution, jump
          events) with abnormal-track flags.
  * AC2 — carried through from the run's tracking-quality block.

    python -m ritsms.validate [--out-dir ritsms_out] [--fps 15] [--samples 8]
"""

import os
import sys
import glob
import json
import math
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ritsms.validate")

WARN_TTC = 3.0
CRIT_TTC = 1.5


def _newest(out_dir, pattern):
    files = sorted(glob.glob(os.path.join(out_dir, pattern)), key=os.path.getmtime)
    return files[-1] if files else None


def _circ_mean_deg(vals):
    r = np.radians(np.asarray(vals, dtype=np.float64))
    return math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean()))


def _movement_label(bearings) -> str:
    """Classify a track's movement from its heading change over its lifetime."""
    b = [x for x in bearings if not (x is None or (isinstance(x, float) and math.isnan(x)))]
    if len(b) < 4:
        return "unknown"
    k = max(2, len(b) // 5)
    start = _circ_mean_deg(b[:k])
    end = _circ_mean_deg(b[-k:])
    delta = (end - start + 180.0) % 360.0 - 180.0
    a = abs(delta)
    if a < 30.0:
        return "through"
    if a >= 150.0:
        return "u_turn"
    return "left" if delta > 0 else "right"   # sign = heading CCW+ in BEV plane


# ── AC4: 15-minute volume + movement counts ───────────────────────────────

def ac4_volume_counts(traj_df, fps, out_dir):
    rows = []
    for oid, g in traj_df.groupby("ObjectID"):
        g = g.sort_values("FrameNumber")
        first_frame = int(g["FrameNumber"].iloc[0])
        cls = str(g["ObjectType"].iloc[0])
        mv = _movement_label(list(g["Bearing"].values))
        win = int((first_frame / fps) // 900)      # 15 min = 900 s
        rows.append({"window": f"{win*15}-{(win+1)*15}min", "class": cls,
                     "movement": mv, "ObjectID": oid})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    counts = (df.groupby(["window", "class", "movement"])["ObjectID"]
                .nunique().reset_index().rename(columns={"ObjectID": "count"}))
    csv_path = os.path.join(out_dir, "ac4_volume_counts_15min.csv")
    counts.to_csv(csv_path, index=False)

    # stacked bar: class on x, movement stacked (first window shown; others in csv)
    w0 = counts["window"].iloc[0]
    sub = counts[counts["window"] == w0]
    piv = sub.pivot_table(index="class", columns="movement", values="count",
                          aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    piv.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(f"AC4 — road-user volume by class & movement ({w0})")
    ax.set_ylabel("unique tracks"); ax.set_xlabel("class")
    ax.legend(title="movement", fontsize=8)
    fig.tight_layout()
    png = os.path.join(out_dir, "ac4_volume_counts_15min.png")
    fig.savefig(png, dpi=110); plt.close(fig)
    logger.info("AC4 -> %s  (%d class/movement rows)", csv_path, len(counts))
    return {"csv": csv_path, "png": png, "total_tracks": int(df["ObjectID"].nunique())}


# ── AC6: TTC/PET-vs-frame plots per sampled conflict ──────────────────────

def _sample_conflicts(conflicts, n):
    crit = [c for c in conflicts if c.get("level") == "CRITICAL"]
    warn = [c for c in conflicts if c.get("level") == "WARNING"]
    ordered, seen = [], set()
    for pool in (crit, warn):                     # criticals first, diversify encounter
        for c in sorted(pool, key=lambda x: x.get("encounter_type", "")):
            enc = c.get("encounter_type")
            if enc not in seen or len(ordered) < n:
                ordered.append(c); seen.add(enc)
            if len(ordered) >= n:
                return ordered[:n]
    return ordered[:n]


def ac6_ttc_plots(conflicts, traces_df, fps, out_dir, n=8, window_s=3.0):
    if traces_df is None or traces_df.empty or not conflicts:
        logger.warning("AC6: no traces/conflicts — skipped.")
        return None
    wf = int(window_s * fps)
    made = []
    traces_df = traces_df.copy()
    traces_df["pair"] = traces_df.apply(lambda r: frozenset((int(r.id1), int(r.id2))), axis=1)
    for c in _sample_conflicts(conflicts, n):
        pair = frozenset((int(c["roaduser1_id"]), int(c["roaduser2_id"])))
        cf = int(c["frame"])
        measure = "ttc" if (c.get("governing_measure") == "TTC" or c.get("ttc_s") is not None) else "pet"
        sub = traces_df[(traces_df["pair"] == pair) &
                        (traces_df["frame"] >= cf - wf) & (traces_df["frame"] <= cf + wf)]
        sub = sub.dropna(subset=[measure]).sort_values("frame")
        if len(sub) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(sub["frame"], sub[measure], "-o", ms=3, color="#4c8bf5", label=measure.upper())
        if measure == "ttc":
            ax.axhline(WARN_TTC, color="#ff8c00", ls="--", lw=1, label="WARNING 3 s")
            ax.axhline(CRIT_TTC, color="#ff4444", ls="--", lw=1, label="CRITICAL 1.5 s")
        ax.axvline(cf, color="#888", ls=":", lw=1, label="conflict frame")
        ax.axvspan(cf - int(0.5 * fps), cf, color="#ffe0b0", alpha=0.35, label="NFb3 0.5 s window")
        ax.set_xlabel("frame"); ax.set_ylabel(f"{measure.upper()} (s)")
        # NFb3 is a TTC-specific validity rule (§5.6) — omit it on PET plots,
        # where it is not applicable, to avoid misleading the reviewer.
        suffix = f" NFb3={c.get('nfb3','?')}/{int(0.5*fps)}" if measure == "ttc" else ""
        ax.set_title(f"AC6 — {c.get('encounter_type')} {tuple(sorted(pair))} "
                     f"[{c.get('level')}]{suffix}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        p = os.path.join(out_dir, f"ac6_{measure}_{'_'.join(map(str, sorted(pair)))}_f{cf}.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        made.append(p)
    logger.info("AC6 -> %d TTC/PET-vs-frame plots", len(made))
    return made


# ── AC3: trajectory integrity ─────────────────────────────────────────────

def ac3_trajectory_integrity(traj_df, fps, min_age, jitter_step_m, max_speed, out_dir):
    stats, abnormal = [], []
    for oid, g in traj_df.groupby("ObjectID"):
        g = g.sort_values("FrameNumber")
        life = int(g["FrameNumber"].iloc[-1] - g["FrameNumber"].iloc[0] + 1)
        xy = g[["BEV_X", "BEV_Y"]].values
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.array([0.0])
        max_jump = float(steps.max()) if steps.size else 0.0
        max_sp = float(g["Speed"].max())
        stats.append({"ObjectID": oid, "lifespan": life, "max_jump_m": round(max_jump, 2),
                      "max_speed": round(max_sp, 2)})
        flags = []
        if life < min_age:
            flags.append("short_track")
        if max_jump > jitter_step_m:
            flags.append("position_jump")
        if max_sp > max_speed:
            flags.append("implausible_speed")
        if flags:
            abnormal.append({"ObjectID": int(oid), "flags": flags,
                             "lifespan": life, "max_jump_m": round(max_jump, 2)})
    sdf = pd.DataFrame(stats)
    summary = {
        "unique_tracks": int(len(sdf)),
        "median_lifespan_frames": int(sdf["lifespan"].median()) if len(sdf) else 0,
        "short_tracks_below_min_age": int((sdf["lifespan"] < min_age).sum()) if len(sdf) else 0,
        "tracks_with_position_jump": int((sdf["max_jump_m"] > jitter_step_m).sum()) if len(sdf) else 0,
        "speed_p50": round(float(traj_df["Speed"].median()), 2),
        "speed_p95": round(float(traj_df["Speed"].quantile(0.95)), 2),
        "abnormal_tracks": abnormal[:100],
        "note": "Speeds are BEV-plane units; absolute-speed verification is conditional "
                "on metric calibration (AC5).",
    }
    with open(os.path.join(out_dir, "ac3_trajectory_integrity.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if len(sdf):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(sdf["lifespan"], bins=30, color="#4c8bf5"); ax[0].set_title("Track lifespan (frames)")
        ax[0].axvline(min_age, color="#ff4444", ls="--", label=f"min_age {min_age}"); ax[0].legend()
        ax[1].hist(traj_df["Speed"].clip(upper=traj_df["Speed"].quantile(0.99)), bins=30, color="#37c871")
        ax[1].set_title("Speed distribution (BEV units)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ac3_trajectory_integrity.png"), dpi=110)
        plt.close(fig)
    logger.info("AC3 -> %d tracks, %d abnormal", summary["unique_tracks"], len(abnormal))
    return summary


def run(out_dir, fps, samples):
    report_p = _newest(out_dir, "report_*.json")
    traces_p = _newest(out_dir, "traces_*.csv")
    traj_p = _newest(out_dir, "trajectory_*.csv")
    if not (report_p and traj_p):
        raise SystemExit(f"Need report_*.json and trajectory_*.csv in {out_dir}. Run the pipeline first.")
    logger.info("report=%s traces=%s traj=%s", os.path.basename(report_p),
                os.path.basename(traces_p) if traces_p else None, os.path.basename(traj_p))

    report = json.load(open(report_p, encoding="utf-8"))
    conflicts = report.get("conflicts", [])
    traj_df = pd.read_csv(traj_p)
    traces_df = pd.read_csv(traces_p) if traces_p else None

    vts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vdir = os.path.join(out_dir, f"validation_{vts}")
    os.makedirs(vdir, exist_ok=True)

    # config values for AC3 thresholds — read from the ritsms Config defaults.
    from ritsms.config import Config
    cfg = Config()

    ac4 = ac4_volume_counts(traj_df, fps, vdir)
    ac6 = ac6_ttc_plots(conflicts, traces_df, fps, vdir, n=samples)
    ac3 = ac3_trajectory_integrity(traj_df, fps, cfg.min_track_age, cfg.jitter_step_m,
                                   cfg.max_speed_mps, vdir)

    pack = {
        "generated_at": datetime.now().isoformat(),
        "source_report": os.path.basename(report_p),
        "AC2_tracking_quality": report.get("tracking_quality"),
        "AC3_trajectory_integrity": ac3,
        "AC4_volume_counts": ac4,
        "AC6_ttc_plots": [os.path.basename(p) for p in (ac6 or [])],
        "calibration_confidence": report.get("calibration_confidence"),
        "note_AC1_AC5": "AC1 RMSE/P95 and AC5 absolute-speed verification require "
                        "metric (lat/lng GCP) calibration; site is currently "
                        f"'{report.get('calibration_confidence')}'.",
    }
    with open(os.path.join(vdir, "acceptance_pack.json"), "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)
    logger.info("Acceptance pack -> %s", vdir)
    return vdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="ritsms_out")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()
    run(args.out_dir, args.fps, args.samples)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
