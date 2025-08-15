from ultralytics import YOLO

model = YOLO("yolov8n.pt") # YOLOv8 nano, small and fast for rasp pi
# .pt = pretrained weights, the model learned from COCO dataset (80 object types)
results = model.predict(source=0, imgsz=640, conf=0.5, show=True, save=True, project="week1_tests") # source 0 means webcam