from pathlib import Path

import torch
from ultralytics import YOLO


def exort():
    this_dir = Path(__file__).parent
    yolo_best_pt_path = this_dir.joinpath("runs/detect/train/weights/best.pt")
    YOLO(str(yolo_best_pt_path)).export(format="onnx")


if __name__ == '__main__':

    def select_device():
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'


    print(select_device())
    # exort()
