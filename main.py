from ultralytics import YOLO

model=YOLO('yolov8n.yaml')
model.train(
    data='/root/data.yaml',
    epochs=150,
    imgsz=512,
    batch=8,
    workers=0,
    pretrained=False,
    iou=0.6,
    device='mps',
    # Knowledge Distillation Parameters
    Mode='KD', # [KD / Training]
    KD_Method='FGD', # ['AFD' / 'ReviewKD' / 'OST' / 'CrossKD' / 'FGD']  已集成的知识蒸馏方法
    teacher='TrainedModels/.../yolov8s.pt', # 预训练好的教师模型，请换成自己本地教师模型路径
    model_pt_path="PretrainedModelFromOfficial/yolov8s.pt", # 来自官方的预训练权重，具体使用方法参见ultralytics/engine/trainer.py line 348
    alpha=0.01, # weight for distillation loss
)

model.val(data="/root/data.yaml", imgsz=512, batch=8, iou=0.6, split='val',device='mps')

#   不同KD方法的最优参数
#    KD       |      alpha
#   AFD       |       100
# ReviewKD    |       30
#   OST       |       100
# CrossKD     |       200
#   FGD       |       0.01
