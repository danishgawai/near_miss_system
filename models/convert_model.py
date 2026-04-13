from ultralytics import YOLO
import os
# Define the path to your PyTorch model
MODEL_PATH = 'yolov8s_merger8_exp1.pt'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# Define the input image size
IMG_SIZE = 640

def main():
    """
    Loads the YOLOv8 model and exports it to ONNX, OpenVINO, and TensorRT formats.
    """
    print(f"Loading YOLOv8 model from {MODEL_PATH}...")
    # Load the YOLOv8 model
    model = YOLO(MODEL_PATH)
    
    # --- Export to ONNX ---
    # This is often a prerequisite for other formats.
    # opset=12 is a good choice for compatibility.
    # print("\nExporting to ONNX format...")
    # model.export(
    #     format='onnx',
    #     half=True,
    #     imgsz=IMG_SIZE,
    #     opset=12,
    #     simplify=True, # Creates a simpler graph for better optimization
    # )
    # print("✅ ONNX export complete.")

    # --- Export to OpenVINO (for CPU) ---
    # The 'half=False' flag ensures it uses full-precision floating-point numbers (FP32),
    # which is generally best for CPUs.
    print("\nExporting to OpenVINO format for CPU...")
    model.export(
        format='openvino',
        imgsz=IMG_SIZE,
        # half=True, # Use FP32 for CPU
        int8=True
    )
    print("✅ OpenVINO export complete.")

    # --- Export to TensorRT (for NVIDIA GPU) ---
    # The 'half=True' flag enables FP16 precision, which significantly speeds up
    # inference on modern NVIDIA GPUs with minimal accuracy loss.
    # print("\nExporting to TensorRT format for GPU...")
    # model.export(
    #     format='engine', # 'engine' is the alias for TensorRT
    #     imgsz=IMG_SIZE,
    #     half=True, # Use FP16 for GPU
    #     device=0,  # Specify the GPU device index
    # )
    # print("✅ TensorRT export complete.")
    print("\nAll conversions finished successfully!")


if __name__ == '__main__':
    main()