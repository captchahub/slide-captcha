from pathlib import Path

import cv2
import numpy as np
import onnxruntime


def yolo_postprocess(outputs, confidence_threshold=0.5, nms_threshold=0.4):
    # 1. 提取置信度
    object_confidences = outputs[:, 4]
    confident_indices = object_confidences > confidence_threshold
    confident_detections = outputs[confident_indices]

    # 2. 提取每个检测的最高分数的类别
    class_scores = confident_detections[:, 5:]
    detected_classes = class_scores.argmax(axis=1)
    class_confidences = class_scores.max(axis=1)

    # 3. 提取坐标
    boxes = confident_detections[:, :4]
    # 注意: 这里的坐标格式是[x_center, y_center, width, height]
    # 您可能需要转换为[x_min, y_min, x_max, y_max]，以便后续的NMS

    # 4. 执行NMS
    nms_indices = cv2.dnn.NMSBoxes(boxes.tolist(), class_confidences.tolist(), confidence_threshold, nms_threshold)
    nms_indices = nms_indices.flatten()

    # 提取NMS后的检测框及相关信息
    final_boxes = boxes[nms_indices]
    final_class_scores = detected_classes[nms_indices]
    final_class_confidences = class_confidences[nms_indices]

    return final_boxes, final_class_scores, final_class_confidences


if __name__ == '__main__':
    this_dir = Path(__file__).parent
    model = this_dir.joinpath("runs/detect/train/weights/best.onnx")

    session = onnxruntime.InferenceSession(model)

    image = cv2.imread("valid/images/0.png")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640))  # YOLO输入的期望宽度和高度
    image_normalized = image_resized.astype(np.float32) / 255.0  # 归一化
    input_data = image_normalized.transpose(2, 0, 1)  # 改变维度从 HWC 到 CHW
    input_data = np.expand_dims(input_data, axis=0)  # 添加批次维度

    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)

    print(yolo_postprocess(outputs[0]))
