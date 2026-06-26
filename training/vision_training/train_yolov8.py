from ultralytics import YOLO


model = YOLO("yolov8n.pt")
model.train(
    data="training/vision_training/dataset.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    project="runs/fairy-vision",
    name="yolov8n-custom",
)
model.export(format="onnx", imgsz=640, simplify=True)
