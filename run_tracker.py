__author__ = "Danish Ahmed"
__copyright__ = ""
__credits__ = ["Danish Ahmed"]
__license__ = ""
__version__ = "0.1.0" 
__maintainer__ = "Danish Ahmed Gawai"
__email__ = "danishh163@gmail.com"
__status__ = "Development"
__module_name__ = "Main"

import sys
import logging
import cv2
import time
import numpy as np
import traceback
from collections import defaultdict
from datetime import datetime

# from utils.draw_utils import draw_text, draw_poly, draw_bb
from utils.yolo_infer import YOLO_Inference
from tracker.byte_tracker import BYTETracker


source_stream = "vehicle_intersection.mp4"
maxDisappeared = 20
ROI = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]
# FPS = 3

# Initialize detector with explicit device
detector = YOLO_Inference(
    model_path="models/yolo26n_int8_openvino_model",
    conf_thres=0.5,
    device="cpu", # OpenVINO usually runs on CPU in Colab unless using an iGPU
    filter_class_ids=[2, 3, 5, 7]  # Filter for Car, Bus, Truck, Motorcycle
)

print("Warming up the OpenVINO model for 10 frames to optimize compilation...")
dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
for i in range(10):
    _ = detector.infer(dummy_frame)
print("Warm-up complete. Model is now optimized.")

logging.info("Starting processing on RTSP")


def plot_detections(frame, detections):
    if detections:
        for (x1, y1, x2, y2, conf_percent, cls_id) in detections:
            label = f"ClsID {cls_id}: {conf_percent}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 100), 2)
            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_bg_y1 = max(y1 - text_h - baseline - 3, 0)
            text_bg_y2 = y1 - baseline + 3
            cv2.rectangle(frame, (x1, text_bg_y1), (x1 + text_w + 4, text_bg_y2), (200, 0, 100), -1)
            cv2.putText(frame, label, (x1 + 2, text_bg_y2 - baseline - 3), font, font_scale, (255, 255, 255), thickness)
    return frame


def predict_on_RTSP():
    cap = cv2.VideoCapture(source_stream)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {source_stream}")
        return
    
    time_string = str(datetime.now()).replace(" ", "_")
    output_video_path = f"video_out_{time_string}.mp4"
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = FPS  # force FPS
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30.0
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    tracker = BYTETracker(
        track_thresh=0.5,
        track_buffer=maxDisappeared,
        match_thresh=0.8,
        frame_rate=float(fps),
    )
    frame_count, total_dets, total_time = 0, 0, 0.0
    track_history = defaultdict(list)
    run_time = 10000  # seconds
    initialize_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        loop_start_time = time.time()
        frame_count += 1
        annotated_frame = frame.copy()
        detections = []

        # Detection
        try:
            boxes = detector.infer(annotated_frame)
            print(boxes)
            # for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
            #     detections.append((x1, y1, x2, y2, round(score * 100, 1), cls_id))
        except Exception as e:
            logging.critical(f"{__module_name__} Detector failed. Error: {e}")
            logging.debug(f"{__module_name__}:\n {traceback.format_exc()}")

        # Tracking
        try:
            centroids_boxes = tracker.update(np.array(boxes).astype(np.float32))
        except Exception as e:
            logging.critical(f"{__module_name__} Error in Tracking: {e}")
            logging.debug(f'{__module_name__}: \n {traceback.format_exc()}')
            time.sleep(2)
            continue

        for centroid_id, centroid, box, score, _ in centroids_boxes:
            box = box.astype(np.int32)
            # annotated_frame = draw_text(annotated_frame, [centroid[0] - 10, centroid[1] - 10], f"{int(score)}% - {centroid_id}")
            # annotated_frame = draw_bb(annotated_frame, [list(box)])
            track = track_history[centroid_id]
            track.append((centroid[0], centroid[1]))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        # Overlay detections
        annotated_frame = plot_detections(annotated_frame, detections)
        out.write(annotated_frame)

        # Stats
        num_dets_frame = len(detections)
        total_dets += num_dets_frame
        loop_time = time.time() - loop_start_time
        total_time += loop_time
        avg_fps_overall = frame_count / total_time if total_time > 0 else float('inf')
        print(f"Frame: {frame_count}, Dets: {num_dets_frame}, Time: {loop_time*1000:.2f}ms, Avg FPS: {avg_fps_overall:.2f}")

        if (time.time() - initialize_time) > run_time:
            print("Stopping, saving video...")
            break

    cap.release()
    out.release()


if __name__ == "__main__":
    print("Starting Processing for RTSP")
    try:
        predict_on_RTSP()
    except KeyboardInterrupt:
        print("Interrupted by user, saving video...")
        sys.exit(0)
    except Exception as e:
        print(f"Code crashed unexpectedly: {e}")
        print(traceback.format_exc())
    print("Processing Completed for RTSP")
