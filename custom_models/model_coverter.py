from ultralytics import YOLO

# Load your model
model = YOLO("C:/Users/sanyk/Downloads/SHAI/yolo-object-detection-onnxruntime-web/custom_models/yolov8n.pt")

# Export to ONNX
model.export(format="onnx", opset=12, dynamic=True)