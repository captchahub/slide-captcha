from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image

yolo_classes = ["target"]


def parse_row(row):
    xc, yc, w, h = row[:4]
    x1 = (xc - w / 2) / 640 * img_width
    y1 = (yc - h / 2) / 640 * img_height
    x2 = (xc + w / 2) / 640 * img_width
    y2 = (yc + h / 2) / 640 * img_height
    prob = row[4:].max()
    class_id = row[4:].argmax()
    label = yolo_classes[class_id]
    return [x1, y1, x2, y2, label, prob]


def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)


def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)


if __name__ == '__main__':
    this_dir = Path(__file__).parent
    model_path = this_dir.joinpath("runs/detect/train/weights/best.onnx")

    model = onnxruntime.InferenceSession(model_path)

    # inputs = model.get_inputs()
    #
    # print(len(inputs))
    #
    # input = inputs[0]
    #
    # print("Name:", input.name)
    # print("Type:", input.type)
    # print("Shape:", input.shape)

    image_path = this_dir.joinpath("train/images/1.png")
    img = Image.open(image_path)
    img_width, img_height = img.size
    img = img.resize((640, 640))

    img = img.convert("RGB")

    input = np.array(img)

    input = input.transpose(2, 0, 1)

    input = input.reshape(1, 3, 640, 640)

    input = input / 255.0

    input = input.astype(np.float32)

    # outputs = model.get_outputs()
    # output = outputs[0]
    # print("Name:", output.name)
    # print("Type:", output.type)
    # print("Shape:", output.shape)
    outputs = model.run(["output0"], {"images": input})

    output = outputs[0]

    output = output.transpose()

    boxes = [row for row in [parse_row(row) for row in output] if row[5] > 0.5]

    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < 0.7]

    print(result[0])
