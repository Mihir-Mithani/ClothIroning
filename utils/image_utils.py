"""
Image Utilities
Helper functions for resize, restore, comparison, and mask visualization.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def resize_for_processing(
    image: Image.Image,
    max_size: int = 1024,
) -> tuple[Image.Image, dict]:
    """
    Resize image so the longest side ≤ max_size, maintaining aspect ratio.
    Returns the resized image and info needed to restore the original size.
    """
    w, h    = image.size
    scale   = min(max_size / max(w, h), 1.0)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    return resized, {"original_w": w, "original_h": h, "scale": scale}


def restore_original_size(
    processed: Image.Image,
    original: Image.Image,
    scale_info: dict,
) -> Image.Image:
    """Upscale processed image back to the original dimensions."""
    ow, oh = scale_info["original_w"], scale_info["original_h"]
    if processed.size == (ow, oh):
        return processed
    return processed.resize((ow, oh), Image.LANCZOS)


def create_comparison_image(
    original: Image.Image,
    ironed: Image.Image,
    divider_width: int = 4,
    label_height: int = 36,
) -> np.ndarray:
    """
    Create a side-by-side before/after comparison image with labels.
    """
    w, h = original.size
    ironed_resized = ironed.resize((w, h), Image.LANCZOS)

    canvas_w = w * 2 + divider_width
    canvas_h = h + label_height

    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))

    # paste images
    canvas.paste(original,       (0,                 label_height))
    canvas.paste(ironed_resized, (w + divider_width, label_height))

    # divider line
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [w, 0, w + divider_width - 1, canvas_h],
        fill=(255, 255, 255),
    )

    # labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    label_bg_before = (40, 40, 40)
    label_bg_after  = (29, 158, 117)   # teal

    # before label
    draw.rectangle([0, 0, w, label_height - 1], fill=label_bg_before)
    draw.text((12, 10), "BEFORE", fill=(255, 255, 255), font=font)

    # after label
    draw.rectangle([w + divider_width, 0, canvas_w, label_height - 1], fill=label_bg_after)
    draw.text((w + divider_width + 12, 10), "AFTER (ironed)", fill=(255, 255, 255), font=font)

    return np.array(canvas)


def overlay_wrinkle_mask(
    image: Image.Image,
    wrinkle_map: np.ndarray,
    garment_mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay a heatmap of wrinkle intensity on the image.
    Red = high wrinkle, yellow = medium, transparent = low.
    """
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]

    # Ensure map matches image size
    if wrinkle_map.shape != (h, w):
        wrinkle_map = cv2.resize(wrinkle_map, (w, h), interpolation=cv2.INTER_LINEAR)
    if garment_mask.shape != (h, w):
        garment_mask_rs = cv2.resize(
            garment_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        garment_mask_rs = garment_mask

    # apply garment mask to wrinkle map
    viz_map = wrinkle_map * garment_mask_rs.astype(float)

    # normalize
    viz_map = (viz_map / (viz_map.max() + 1e-6) * 255).astype(np.uint8)

    # colourmap: COLORMAP_JET gives blue→green→yellow→red
    heat = cv2.applyColorMap(viz_map, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # only show heatmap where wrinkle_map > threshold
    threshold_mask = (viz_map > 40).astype(np.float32)[..., np.newaxis]
    blended = (
        heat.astype(float) * threshold_mask * alpha
        + arr.astype(float) * (1.0 - threshold_mask * alpha)
    ).clip(0, 255).astype(np.uint8)

    # draw garment contour
    contour_mask = garment_mask_rs.astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 150), 2)

    return blended
