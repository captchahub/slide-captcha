from pathlib import Path

this_dir = Path(__file__).parent

# template
template_root_path = this_dir.joinpath("template")
template_width = 60

# background
background_root_path = this_dir.joinpath("background")
background_width = 300
background_height = 200

background_size = 300

# yolo


yolo_train_num = 3000
yolo_valid_num = 100

yolo_path = this_dir.joinpath("../yolo")

yolo_train_path = f"{yolo_path}/train"
yolo_valid_path = f"{yolo_path}/valid"

yolo_train_image_path = f"{yolo_train_path}/images"
yolo_train_labels_path = f"{yolo_train_path}/labels"

yolo_valid_image_path = f"{yolo_valid_path}/images"
yolo_valid_labels_path = f"{yolo_valid_path}/labels"
