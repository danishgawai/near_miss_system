"""§4.3 — Multi-object tracking (ByteTrack) + tracking-quality monitor (AC2).

Wraps the vendored ByteTrack. The quality monitor reports the three proposal
indicators: identity switches (logged for annotated review — a true count needs
ground truth), track fragmentation (GT-free proxy: a new track appearing next
to a just-dead one), and jump/jitter rate (from the trajectory Kalman's outlier
events). Numeric pass/fail thresholds are intentionally NOT frozen here.
"""

import logging
import numpy as np
from tracker.byte_tracker import BYTETracker

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, cfg, fps):
        self.cfg = cfg
        self.bt = BYTETracker(track_thresh=cfg.track_thresh, track_buffer=cfg.track_buffer,
                              match_thresh=cfg.match_thresh, frame_rate=float(fps))

    def update(self, dets, w, h):
        arr = np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)
        raw = self.bt.update(arr)
        max_w, max_h = w * self.cfg.track_max_box_ratio, h * self.cfg.track_max_box_ratio
        out = []
        for tid, centroid, box, score, cls_id in raw:
            x1, y1, x2, y2 = box
            if (x2 - x1) <= max_w and (y2 - y1) <= max_h:
                out.append((int(tid), box, float(score), int(cls_id)))
        return out


class TrackQualityMonitor:
    """Feed per-frame {track_id: bev_position}. Accumulates AC2 indicators."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._prev_ids = set()
        self._recent_dead = {}          # id -> (last_frame, last_pos)
        self.births = 0
        self.deaths = 0
        self.fragments = 0
        self.jumps = 0
        self.frames = 0
        self._seen_ids = set()

    def update(self, frame_idx, positions: dict, jump_delta: int = 0):
        self.frames += 1
        self.jumps += int(jump_delta)
        ids = set(positions)
        born = ids - self._prev_ids
        died = self._prev_ids - ids
        self.births += len(born)
        self.deaths += len(died)
        self._seen_ids |= ids

        # fragmentation proxy: a newborn appearing near a recently-dead track
        for b in born:
            pb = np.asarray(positions[b], dtype=np.float64)
            for did, (df, dp) in list(self._recent_dead.items()):
                if (frame_idx - df) <= self.cfg.frag_reassoc_gap_frames and \
                        np.linalg.norm(pb - dp) <= self.cfg.frag_reassoc_dist_m:
                    self.fragments += 1
                    break
        for d in died:
            # store last known position from the previous frame's set if we had it
            self._recent_dead[d] = (frame_idx, self._prev_pos.get(d, np.zeros(2))) \
                if hasattr(self, "_prev_pos") else (frame_idx, np.zeros(2))
        # expire old dead records
        self._recent_dead = {i: v for i, v in self._recent_dead.items()
                             if frame_idx - v[0] <= self.cfg.frag_reassoc_gap_frames}
        self._prev_ids = ids
        self._prev_pos = {i: np.asarray(p, dtype=np.float64) for i, p in positions.items()}

    def summary(self) -> dict:
        return {
            "unique_tracks": len(self._seen_ids),
            "track_births": self.births,
            "track_deaths": self.deaths,
            "fragmentation_proxy": self.fragments,
            "jump_jitter_events": self.jumps,
            "frames": self.frames,
            "note": "IDSW requires annotated ground truth (AC2 manual review); "
                    "fragmentation is a GT-free proxy; thresholds set post-baseline.",
        }
