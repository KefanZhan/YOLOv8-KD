from ultralytics import YOLO

model=YOLO('yolov8n.yaml')
model.train(
    data='NWPU/data.yaml',
    epochs=150,
    imgsz=512,
    batch=8,
    workers=0,
    pretrained=False,
    iou=0.6,
    device='mps',
    # Knowledge Distillation Parameters
    Mode='KD', # [KD / Training]
    KD_Method='FGD', # ['AFD' / 'ReviewKD' / 'OST' / 'CrossKD' / 'FGD']
    teacher='TrainedModels/NWPU/yolov8s.pt',
    model_pt_path="PretrainedModelFromOfficial/yolov8s.pt",
    alpha=0.01, # weight for distillation loss
)

#   不同KD方法的最优参数
#    KD       |      alpha
#   AFD       |       100
# ReviewKD    |       30
#   OST       |       100
# CrossKD     |       200
#   FGD       |       0.01
