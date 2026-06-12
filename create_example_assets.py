"""
Generate placeholder example images for the Gradio UI examples panel.
Run once before publishing to HuggingFace Spaces.

  python create_example_assets.py
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXAMPLES = [
    ("example_shirt.jpg",    "Shirt",    (200, 210, 230)),
    ("example_tshirt.jpg",   "T-Shirt",  (230, 200, 200)),
    ("example_dress.jpg",    "Dress",    (200, 230, 210)),
    ("example_trousers.jpg", "Trousers", (210, 200, 230)),
]


def draw_wrinkle_lines(draw, x0, y0, x1, y1, color, n=12):
    """Draw random curved wrinkle lines within a bounding box."""
    import random
    rng = random.Random(42)
    for _ in range(n):
        y = rng.randint(y0, y1)
        pts = []
        for xi in range(x0, x1, 4):
            yi = y + rng.randint(-6, 6)
            pts.append((xi, yi))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=1)


def create_placeholder(filename: str, label: str, color: tuple):
    W, H = 400, 600
    img  = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # body silhouette
    draw.ellipse([130, 30, 270, 140],   fill=(210, 180, 140))   # head
    draw.rectangle([120, 140, 280, 420], fill=color)             # torso / shirt
    draw.rectangle([120, 420, 195, 590], fill=(180, 160, 120))  # left leg
    draw.rectangle([205, 420, 280, 590], fill=(180, 160, 120))  # right leg
    draw.rectangle([60,  150, 120, 330], fill=color)             # left arm
    draw.rectangle([280, 150, 340, 330], fill=color)             # right arm

    # wrinkle lines on shirt
    wrinkle_color = tuple(max(c - 40, 0) for c in color)
    draw_wrinkle_lines(draw, 125, 160, 275, 410, wrinkle_color, n=18)
    draw_wrinkle_lines(draw, 65,  155, 115, 325, wrinkle_color, n=8)
    draw_wrinkle_lines(draw, 285, 155, 335, 325, wrinkle_color, n=8)

    # label
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, W, 36], fill=(40, 40, 40))
    draw.text((10, 8), f"Example: {label} (wrinkled)", fill=(255, 255, 255), font=font)

    out_path = os.path.join("assets", filename)
    img.save(out_path, quality=92)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    for fname, lbl, col in EXAMPLES:
        create_placeholder(fname, lbl, col)
    print("Done — example assets saved to assets/")
