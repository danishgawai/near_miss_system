__author__ = "Danish Ahmed"
__copyright__ = ""
__credits__ = ["Danish Ahmed"]
__license__ = ""
__version__ = "0.1.0" 
__maintainer__ = "Danish Ahmed Gawai"
__email__ = "danishh163@gmail.com"
__status__ = "Development"
__module_name__ = "Yolov8 Inference"

# import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class YOLO_Inference:
    def __init__(self, model_path, device=None, conf_thres=0.25, filter_class_ids=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thres = conf_thres
        self.filter_class_ids = filter_class_ids

        # Load YOLOv8 model (ultralytics wrapper)
        self.model = YOLO(model_path,task='detect') #.to(self.device)
        # self.model.predict(device="intel:cpu")
    def infer(self, img_bgr):
        """Run YOLOv8 inference on a single image (OpenCV format)."""
        # start_time = time.time()
        results = self.model.predict(img_bgr, conf=self.conf_thres, device=self.device, verbose=False, classes=self.filter_class_ids)
        # print(f"Inference time: {(time.time() - start_time)*1000:.2f} ms")
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)   # [x1,y1,x2,y2]
        scores = res.boxes.conf.cpu().numpy()              # float
        class_ids = res.boxes.cls.cpu().numpy().astype(int)
        
        # Convert into desired format
        detections = [
            box.tolist() + [int(score * 100), int(cls_id)]
            for box, score, cls_id in zip(boxes, scores, class_ids)
        ]
    
        return detections

        # return boxes, scores, class_ids

def draw_results(img, boxes, scores, class_ids, class_names=None):
    """Draw bounding boxes on an image."""
    for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
        label = class_names[cls_id] if class_names else str(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


if __name__ == "__main__":
    model_path = "models/yolo26n_openvino_model/"   # path to YOLOv8 model
    img_path = "professional.png"       # path to test image
    class_names = ["Car", "Bus", "Truck", "Motorcycle", "Pedestrian", "Bicycle"]  # replace with your dataset's classes

    detector = YOLO_Inference(model_path,device="cpu", conf_thres=0.5)
    image  = cv2.imread(img_path)
    detections = detector.infer(image)

    print("Detections:", detections)

    img_out = draw_results(image, [d[:4] for d in detections], [d[4] for d in detections], [d[5] for d in detections])
    cv2.imwrite("output.jpg", img_out)

    # cv2.imshow("Result", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
