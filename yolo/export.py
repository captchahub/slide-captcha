from pathlib import Path

from ultralytics import YOLO


def exort():
    this_dir = Path(__file__).parent
    yolo_best_pt_path = this_dir.joinpath("runs/detect/train/weights/best.pt")
    YOLO("runs/detect/train/weights/best.pt").export(format="onnx")


if __name__ == '__main__':
    exort()
