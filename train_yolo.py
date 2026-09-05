from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # lightweight

model.train(
    data="tamper.yaml",
    epochs=50,
    imgsz=512,
    batch=8,
    name="tamper_model"
)