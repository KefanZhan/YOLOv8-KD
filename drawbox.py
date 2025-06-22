import cv2
import numpy as np


def draw_ground_truth_boxes(image_path, label_path):
    # 读取图片
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 定义每个类别的颜色（这里简单定义 10 种颜色，可根据需要扩展）
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255)
    ]

    # 读取标注文件
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 解析标注信息
        class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
        class_id = int(class_id)

        # 将归一化坐标转换为像素坐标
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # 根据类别 ID 选择颜色
        color = colors[class_id % len(colors)]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image


def draw_box(image_path, label_path):
    result_image = draw_ground_truth_boxes(image_path, label_path)

    # 显示结果
    cv2.imshow('Image with Ground Truth Boxes', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite('gt.jpg', result_image)

image_path = "NWPU/images/train/308.jpg"
label_path = "NWPU/labels/train/308.txt"
draw_box(image_path, label_path)