from ultralytics import YOLO

model = YOLO("TrainedModels/NWPU/yolov8m.pt")
model.val(data="", imgsz=512, batch=8, iou=0.6, split='val',device='mps')




#