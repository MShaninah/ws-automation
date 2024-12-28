from PIL import Image, ImageDraw, ImageFont
import random
import os
import numpy as np

# Fonts and Colors
FONT_PATHS = ["arial.ttf", "times.ttf", "cour.ttf", "comic_sans.ttf", "calibri.ttf"]
IMAGE_WIDTH = 280
IMAGE_HEIGHT = 90
FONT_SIZES = range(30, 50)
CHAR_COUNTS = [5, 6]  # Allow for variable-length CAPTCHAs
NOISE_CURVES = 8  # Number of smooth curved lines

# Directories for storing CAPTCHAs
BASE_DIR = "generated_captchas"
TRAIN_DIR = os.path.join(BASE_DIR, "train")

# Ensure directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)

def generate_random_color():
    """Generate a random RGB color."""
    return tuple(random.randint(0, 255) for _ in range(3))

def create_gradient_background():
    """Create a gradient background."""
    base = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), generate_random_color())
    overlay = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), generate_random_color())
    mask = Image.linear_gradient("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))  # Create a gradient mask
    gradient = Image.composite(base, overlay, mask)
    return gradient

def draw_smooth_curved_line(draw):
    """Draw a smooth cubic Bézier curve."""
    for _ in range(NOISE_CURVES):
        start = (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT))
        control1 = (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT))
        control2 = (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT))
        end = (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT))

        t = np.linspace(0, 1, 100)  # Smooth curve
        curve = [
            (
                int((1-t_val)**3 * start[0] + 3*(1-t_val)**2 * t_val * control1[0] + 3*(1-t_val) * t_val**2 * control2[0] + t_val**3 * end[0]),
                int((1-t_val)**3 * start[1] + 3*(1-t_val)**2 * t_val * control1[1] + 3*(1-t_val) * t_val**2 * control2[1] + t_val**3 * end[1])
            )
            for t_val in t
        ]

        draw.line(curve, fill=generate_random_color(), width=random.randint(2, 4))

def add_random_occlusion(draw):
    """Draw random lines or shapes to occlude parts of the CAPTCHA."""
    for _ in range(5):  # Add 5 random occlusions
        x1 = random.randint(0, IMAGE_WIDTH)
        y1 = random.randint(0, IMAGE_HEIGHT)
        x2 = random.randint(0, IMAGE_WIDTH)
        y2 = random.randint(0, IMAGE_HEIGHT)
        draw.line([(x1, y1), (x2, y2)], fill=generate_random_color(), width=random.randint(2, 4))

def draw_rotated_text(image, draw, text, position, font, angle):
    """Draw rotated text on the image."""
    temp_image = Image.new('RGBA', image.size, (255, 255, 255, 0))  # Transparent background
    temp_draw = ImageDraw.Draw(temp_image)
    temp_draw.text(position, text, font=font, fill=generate_random_color())

    rotated_image = temp_image.rotate(angle, resample=Image.BICUBIC, center=position)
    image.paste(Image.alpha_composite(image.convert('RGBA'), rotated_image).convert('RGB'))

def generate_captcha(output_dir, char_count):
    """Generate a single CAPTCHA image."""
    # Use gradient background
    image = create_gradient_background()
    draw = ImageDraw.Draw(image)

    # Add smooth curves and occlusion
    draw_smooth_curved_line(draw)
    add_random_occlusion(draw)

    characters = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=char_count))
    current_x = 10

    for char in characters:
        font_path = random.choice(FONT_PATHS)
        font_size = random.choice(FONT_SIZES)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Font not found: {font_path}. Adjust the FONT_PATHS.")
            return

        text_bbox = draw.textbbox((0, 0), char, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        rotation_angle = random.randint(-60, 60)
        if current_x + text_width + 10 > IMAGE_WIDTH:
            break

        y_position = random.randint(10, IMAGE_HEIGHT - text_height - 10)
        draw_rotated_text(image, draw, char, (current_x, y_position), font, rotation_angle)
        current_x += text_width + random.randint(10, 20)

    file_name = f"{characters}_{random.randint(1000, 9999)}.png"
    file_path = os.path.join(output_dir, file_name)
    image.save(file_path)
    print(f"Generated CAPTCHA saved as {file_path}")

def generate_captchas(num_train):
    """Generate training CAPTCHA dataset."""
    for _ in range(num_train):
        char_count = random.choice(CHAR_COUNTS)
        generate_captcha(TRAIN_DIR, char_count)

NUM_TRAIN_CAPTCHAS = 100
generate_captchas(NUM_TRAIN_CAPTCHAS)