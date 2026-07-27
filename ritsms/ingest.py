"""§4.1 — Video ingestion with FPS normalisation.

Source frame rates vary; downstream persistence windows (NFb3) and derivatives
need a stable temporal basis. This caps the effective processing rate at
``fps_ceiling`` by deterministically sub-sampling, and yields preserved
timestamps so the conflict engine has a consistent Δt.

Yields FramePacket(index, frame, t_s, capture_ts_ms) where index is the
EFFECTIVE (post-subsample) frame counter and t_s = index / fps_eff.
"""

import cv2
import logging
from dataclasses import dataclass
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class FramePacket:
    index: int          # effective frame counter (post subsample)
    frame: object       # BGR ndarray
    t_s: float          # effective video time (seconds), monotonic
    capture_ts_ms: float  # source CAP_PROP_POS_MSEC (CaptureTimeStamp)


class FrameSource:
    def __init__(self, source: str, fps_ceiling: float):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = raw if raw and 1.0 <= raw <= 120.0 else 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_eff = min(self.source_fps, float(fps_ceiling))
        self.stride = max(1, int(round(self.source_fps / self.fps_eff)))
        self.fps_eff = self.source_fps / self.stride   # exact after integer stride
        logger.info("Source %dx%d @ %.2ffps -> stride %d -> fps_eff %.2f (%d frames)",
                    self.width, self.height, self.source_fps, self.stride,
                    self.fps_eff, self.total)

    def frames(self) -> Iterator[FramePacket]:
        src_idx = 0
        eff_idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if src_idx % self.stride == 0:
                eff_idx += 1
                ts_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                yield FramePacket(eff_idx, frame, eff_idx / self.fps_eff, ts_ms)
            src_idx += 1
        self.cap.release()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
