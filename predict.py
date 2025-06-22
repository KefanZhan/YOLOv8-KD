from ultralytics import YOLO

model = YOLO("TrainedModels/Dior/yolov8s.pt")
model.predict(source='',iou=0.4,device='mps',save=True, visualize=True, show_conf=False)