from pathlib import Path

import torch
from ultralytics import YOLO


def select_device():
    if torch.cuda.is_available():
        return 'cuda'

    else:
        return 'cpu'


def train():
    this_dir = Path(__file__).parent
    yolo_last_py_path = this_dir.joinpath("runs/detect/train/weights/last.pt")

    if yolo_last_py_path.exists():
        model = YOLO(str(yolo_last_py_path))
        arg = {
            'resume': True
        }
    else:
        model = YOLO("yolov8n.pt")
        arg = {
            'data': 'data.yaml',
            'epochs': 3,
            'device': select_device()
        }

    model.train(**arg)
    model.val()
    model.export(format="onnx")


if __name__ == '__main__':
    train()
