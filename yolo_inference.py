from ultralytics import YOLO

model = YOLO("yolov8x")

result = model.predict("image.jpg", save=True)