from pathlib import Path

from ultralytics import YOLO


def train():
    this_dir = Path(__file__).parent

    yolo_best_pt_path = this_dir.joinpath("runs/detect/train/weights/best.pt")
    # 加载模型
    # model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

    # 使用模型
    model.train(data="data.yaml", epochs=10, device='mps')  # 训练模型
    metrics = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

    YOLO(str(yolo_best_pt_path)).export(format="onnx")


if __name__ == '__main__':
    train()
