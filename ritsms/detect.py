"""§4.3 — Detection.

Thin backend-agnostic wrapper. Default backend is YOLO (the merger8 domain
model, OpenVINO). An RF-DETR backend can drop in later behind the same
infer() contract. An optional explicit class-agnostic NMS pass prunes
duplicate boxes before tracking (proposal §4.3) when post_nms_iou > 0.

infer(frame) -> list of [x1, y1, x2, y2, score, cls_id].
"""

import logging
import numpy as np
import torch  # Must be imported first to unpack its bundled CUDA DLLs

# ONNXRuntime CUDA DLL preload — only needed for an ONNX backend. It prints
# missing-DLL errors when ORT's CUDA build doesn't match torch's; harmless for
# the .pt/OpenVINO paths, so keep it quiet and non-fatal.
try:
    import onnxruntime as ort
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ort.preload_dlls()
except Exception:
    pass

logger = logging.getLogger(__name__)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _nms(dets, iou_thr):
    order = sorted(dets, key=lambda d: -d[4])
    kept = []
    for d in order:
        if all(_iou(d, k) < iou_thr for k in kept):
            kept.append(d)
    return kept


def _resolve_device(requested: str) -> tuple:
    """Return (device_string, is_gpu). Falls back to CPU with a clear warning
    when a GPU was asked for but torch has no working CUDA — never crashes."""
    req = str(requested).strip().lower()
    if req in ("", "cpu"):
        return "cpu", False
    try:
        import torch
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "cuda:0"
            logger.info("Detector on GPU: %s (torch %s, cuda %s)",
                        name, torch.__version__, torch.version.cuda)
            return (requested if req not in ("gpu", "cuda") else "0"), True
        logger.warning("device='%s' requested but torch has no CUDA (%s). "
                       "Falling back to CPU. Install a CUDA build of torch to use the GPU.",
                       requested, torch.__version__)
    except Exception as e:
        logger.warning("CUDA check failed (%s); using CPU.", e)
    return "cpu", False


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nms_iou = float(cfg.post_nms_iou)
        self.device, self.is_gpu = _resolve_device(cfg.model_device)
        # FP16 only helps on GPUs with fast half-precision (Pascal+). It is
        # slow/emulated on Maxwell, so it stays opt-in via cfg.model_half.
        self.half = getattr(cfg, "model_half", 32) and self.is_gpu
        if self.is_gpu and cfg.model_path.rstrip("/").endswith("openvino_model"):
            logger.warning("model_path is an OpenVINO export — it runs on CPU regardless of "
                           "device. Point --model at the .pt (or a TensorRT .engine) for GPU.")
        if cfg.detector_backend == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(cfg.model_path, task="detect")
            self._filter = cfg.filter_class_ids
        else:
            raise ValueError(f"Unsupported detector_backend: {cfg.detector_backend}")

    def warmup(self, n):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self.infer(dummy)

    def infer(self, frame):
        cfg = self.cfg
        res = self.model.predict(frame, conf=cfg.model_conf, device=self.device,
                                 imgsz=cfg.model_imgsz, quantize=cfg.model_half,
                                 verbose=False, classes=self._filter)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        dets = [list(b) + [float(s), int(c)] for b, s, c in zip(boxes, scores, cls)]
        if self.nms_iou > 0 and dets:
            dets = _nms(dets, self.nms_iou)
        return dets
