from pathlib import Path

import cv2
from ultralytics import YOLO


def train():
    this_dir = Path(__file__).parent

    yolo_best_pt_path = this_dir.joinpath("runs/detect/train/weights/best.pt")

    model = YOLO(str(yolo_best_pt_path))

    results = model("valid/images/1.png")

    # 1. 读取原始图像
    image_path = "valid/images/1.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    train()
