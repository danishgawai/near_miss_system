from typing import List, Tuple, Dict
import numpy as np
from tracker.byte_tracker import BYTETracker


class TrackerManager:
    def __init__(self, track_thresh: float, track_buffer: int, match_thresh: float, fps: float):
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=float(fps),
        )

    def update(self, detections: List[list]):
        arr = np.array(detections, dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)
        return self.tracker.update(arr)

    @staticmethod
    def filter_reasonable_boxes(raw_tracks, frame_w: int, frame_h: int, max_box_ratio: float):
        valid = []
        max_w = frame_w * max_box_ratio
        max_h = frame_h * max_box_ratio
        for t in raw_tracks:
            tid, centroid, box, score, cls_id = t
            x1, y1, x2, y2 = box
            if (x2 - x1) <= max_w and (y2 - y1) <= max_h:
                valid.append(t)
        return valid