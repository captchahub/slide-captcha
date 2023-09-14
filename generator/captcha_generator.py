from PIL import Image, ImageDraw, ImageChops, ImageFilter
import random


def random_int(low, high):
    """Generate a random integer between low and high."""
    return random.randint(low, high)

def generate_captcha_image_v16(background_path, template_path):
    """Generate a captcha image with an adjusted transparency for the notch in the background."""
    background = Image.open(background_path).convert("RGBA")
    template = Image.open(template_path).convert("RGBA")

    max_x = background.width - template.width
    max_y = background.height - template.height
    random_x = random_int(0, max_x)
    random_y = random_int(0, max_y)

    # Extract the notch content from the background
    cutout = background.crop((random_x, random_y, random_x + template.width, random_y + template.height))
    cutout = Image.composite(cutout, Image.new("RGBA", template.size, (0, 0, 0, 0)), template)

    # Apply a lighter and more transparent black notch to the background
    lighter_black_notch = Image.new("RGBA", template.size, (70, 70, 70, 120))  # Lighter and more transparent black
    # lighter_black_notch = lighter_black_notch.filter(ImageFilter.GaussianBlur(2))  # Apply slight blur
    background.paste(lighter_black_notch, (random_x, random_y), template.split()[3])

    # Combine the two images side by side with cutout on the left and aligned vertically
    combined_width = background.width + cutout.width
    combined_height = max(background.height, cutout.height)

    combined_image = Image.new("RGBA", (combined_width, combined_height))
    combined_image.paste(background, (0, 0))
    combined_image.paste(cutout, (0, random_y), cutout)

    return background, cutout, combined_image


# Testing the function


# Testing the functio

if __name__ == '__main__':
    background_path = "1.png"
    fixed_template_path = "p.png"
    # active_template_path = "active.png"
    captcha_image, cutout,c = generate_captcha_image_v16(background_path, fixed_template_path)
    # captcha_image.show()
    # cutout.show()
    c.show()