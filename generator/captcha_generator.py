import asyncio
import os
import random

import httpx
from PIL import Image
from loguru import logger

from generator.config import template_root_path, background_root_path, background_size, template_width, \
    background_width, background_height, yolo_train_num, yolo_valid_num, yolo_train_image_path, yolo_train_labels_path, \
    yolo_valid_image_path, yolo_valid_labels_path, yolo_train_path, yolo_valid_path


def random_int(low, high):
    """Generate a random integer between low and high."""
    return random.randint(low, high)


def generate_captcha_image(background_path, template_path):
    """Generate a captcha image with an adjusted transparency for the notch in the background."""
    background = Image.open(background_path).convert("RGBA")
    template = Image.open(template_path).convert("RGBA")
    template = template.resize((template_width, template_width))

    max_x = background.width - template.width
    max_y = background.height - template.height

    random_x = random_int(template.width, max_x)
    random_y = random_int(0, max_y)

    cutout = background.crop((random_x, random_y, random_x + template.width, random_y + template.height))
    cutout = Image.composite(cutout, Image.new("RGBA", template.size, (0, 0, 0, 0)), template)

    lighter_black_notch = Image.new("RGBA", template.size, (70, 70, 70, 255))  # Lighter and more transparent black
    # lighter_black_notch = lighter_black_notch.filter(ImageFilter.GaussianBlur(2))  # Apply slight blur
    background.paste(lighter_black_notch, (random_x, random_y), template.split()[3])

    # Combine the two images side by side with cutout on the left and aligned vertically
    combined_width = max(background.width, cutout.width)
    combined_height = max(background.height, cutout.height)

    combined_image = Image.new("RGBA", (combined_width, combined_height))
    combined_image.paste(background, (0, 0))
    combined_image.paste(cutout, (0, random_y), cutout)

    # yolo
    x_center = random_x + template.width / 2
    y_center = random_y + template.height / 2

    x_center_normalized = round(x_center / background.width, 3)
    y_center_normalized = round(y_center / background.height, 3)
    width_normalized = round(template.width / background.width, 3)
    height_normalized = round(template.height / background.height, 3)

    return combined_image, (0, x_center_normalized, y_center_normalized, width_normalized, height_normalized)


def get_random_template(path=template_root_path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    return random.choice(files)


def download_image(url, filename):
    with httpx.Client(follow_redirects=True) as client:
        # 使用allow_redirects确保跟随重定向
        response = client.get(url)

        # 确保请求成功
        response.raise_for_status()

        # 使用二进制模式写入文件
        with open(filename, 'wb') as image_file:
            for chunk in response.iter_bytes():
                image_file.write(chunk)


async def async_download_image(client, url, filename):
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        with open(filename, 'wb') as image_file:
            async for chunk in response.aiter_bytes():
                image_file.write(chunk)


async def download_backgrounds():
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [
            async_download_image(
                client,
                f"https://picsum.photos/{background_width}/{background_height}",
                os.path.join(background_root_path, f"{i}.png")
            ) for i in range(background_size)
        ]
        await asyncio.gather(*tasks)


def ensure_yolo_path_exist():
    if not os.path.exists(yolo_train_path):
        os.makedirs(yolo_train_path)

    if not os.path.exists(yolo_valid_path):
        os.makedirs(yolo_valid_path)

    if not os.path.exists(yolo_train_image_path):
        os.makedirs(yolo_train_image_path)

    if not os.path.exists(yolo_train_labels_path):
        os.makedirs(yolo_train_labels_path)

    if not os.path.exists(yolo_valid_image_path):
        os.makedirs(yolo_valid_image_path)

    if not os.path.exists(yolo_valid_labels_path):
        os.makedirs(yolo_valid_labels_path)


if __name__ == '__main__':
    if not os.path.exists(background_root_path):
        os.makedirs(background_root_path)

    background_list = [os.path.join(background_root_path, f) for f in os.listdir(background_root_path)]
    if len(background_list) == 0:
        logger.info("background list is empty, start download background image")
        asyncio.run(download_backgrounds())
        background_list = [os.path.join(background_root_path, f) for f in os.listdir(background_root_path)]

    template_list = [os.path.join(template_root_path, f) for f in os.listdir(template_root_path)]

    ensure_yolo_path_exist()

    for i in range(yolo_train_num):
        background_path = random.choice(background_list)
        template_path = random.choice(template_list)
        combined_image, label = generate_captcha_image(background_path, template_path)
        combined_image.save(f"{yolo_train_image_path}/{i}.png")

        with open(f"{yolo_train_labels_path}/{i}.txt", 'w') as file:
            text = ' '.join(map(str, label))
            file.write(text)

    for i in range(yolo_valid_num):
        background_path = random.choice(background_list)
        template_path = random.choice(template_list)
        combined_image, label = generate_captcha_image(background_path, template_path)
        combined_image.save(f"{yolo_valid_image_path}/{i}.png")

        with open(f"{yolo_valid_labels_path}/{i}.txt", 'w') as file:
            text = ' '.join(map(str, label))
            file.write(text)
