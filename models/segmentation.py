"""
Garment Segmentation Module
Uses SegFormer fine-tuned on the ATR / LIP human-parsing datasets.
Falls back to a simple colour/edge-based approach if GPU is unavailable.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# Clothing label IDs in the mattmdjaga/segformer_b2_clothes palette.
# IMPORTANT: only true garment pixels — body parts (face, arms, legs, hair)
# and accessories (shoes, bag) must NOT be included, or the ironing pipeline
# will smooth skin/face along with the fabric.
CLOTHING_LABELS = {
    "upper_body": [4],        # Upper-clothes (shirt, t-shirt, jacket, coat)
    "lower_body": [6, 5],     # Pants, Skirt
    "dress":      [7],        # Dress
    "extras":     [8, 17],    # Belt, Scarf (worn fabric items)
}

# All valid "garment" ids — used for auto-detect
ALL_GARMENT_IDS = [4, 5, 6, 7, 8, 17]

LABEL_NAMES = {
    0:  "background",
    1:  "hat",
    2:  "hair",
    3:  "sunglasses",
    4:  "upper-clothes",
    5:  "skirt",
    6:  "pants",
    7:  "dress",
    8:  "belt",
    9:  "left-shoe",
    10: "right-shoe",
    11: "face",
    12: "left-leg",
    13: "right-leg",
    14: "left-arm",
    15: "right-arm",
    16: "bag",
    17: "scarf",
}


class GarmentSegmenter:
    """
    Wraps a HuggingFace SegFormer model for human-part segmentation.
    Model: mattmdjaga/segformer_b2_clothes
    """

    MODEL_ID = "mattmdjaga/segformer_b2_clothes"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model      = None
        self._load()

    # ── private ──────────────────────────────────────────────────────────────
    def _load(self):
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            print(f"[Segmenter] Loading {self.MODEL_ID} on {self.device} …")
            self._processor = SegformerImageProcessor.from_pretrained(self.MODEL_ID)
            self._model     = SegformerForSemanticSegmentation.from_pretrained(self.MODEL_ID)
            self._model.to(self.device).eval()
            print("[Segmenter] Model ready.")
        except Exception as exc:
            print(f"[Segmenter] Could not load transformer model: {exc}")
            print("[Segmenter] Falling back to simple masking.")
            self._model = None

    # ── public ────────────────────────────────────────────────────────────────
    def segment(
        self,
        image: Image.Image,
        clothing_type: str = "Auto-detect",
    ) -> dict:
        """
        Parameters
        ----------
        image        : PIL RGB image
        clothing_type: hint for which garment to focus on

        Returns
        -------
        dict with keys:
          garment_mask  – H×W bool ndarray (True = garment pixel)
          labels        – list of detected garment label names
          label_mask    – H×W int ndarray with per-pixel label ids
          confidence    – float 0-1
        """
        if self._model is not None:
            return self._segment_segformer(image, clothing_type)
        return self._segment_fallback(image, clothing_type)

    # ── SegFormer path ────────────────────────────────────────────────────────
    def _segment_segformer(self, image: Image.Image, clothing_type: str) -> dict:
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits          # 1 × C × H' × W'

        # upsample to original size
        h, w = image.size[1], image.size[0]
        upsampled = F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        probabilities = torch.softmax(upsampled, dim=1)
        pred = probabilities.argmax(dim=1).squeeze(0).cpu().numpy()  # H × W

        # decide which labels count as "garment" given the hint
        target_ids = self._target_ids(clothing_type)
        garment_mask = np.isin(pred, target_ids)

        # morphological clean-up (remove tiny islands)
        garment_mask = _clean_mask(garment_mask)

        detected_ids = [int(i) for i in np.unique(pred) if i in target_ids]
        labels       = [LABEL_NAMES.get(i, str(i)) for i in detected_ids]

        pixel_confidence = probabilities.max(dim=1).values.squeeze(0).cpu().numpy()
        confidence = (
            float(pixel_confidence[garment_mask].mean()) if garment_mask.any() else 0.0
        )

        return {
            "garment_mask": garment_mask,
            "labels":       labels,
            "label_mask":   pred,
            "confidence":   confidence,
        }

    # ── Fallback: simple skin-exclusion mask ─────────────────────────────────
    def _segment_fallback(self, image: Image.Image, clothing_type: str) -> dict:
        arr  = np.array(image.convert("RGB")).astype(float)
        h, w = arr.shape[:2]
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

        # exclude skin-coloured pixels (very rough heuristic)
        skin = (r > 95) & (g > 40) & (b > 20) & \
               (r > g) & (r > b) & \
               (np.abs(r - g) > 15) & \
               (r > 100)

        # Estimate background color from image borders and remove pixels close
        # to it. This is still a fallback, but is substantially safer than
        # treating every non-skin, non-white pixel as clothing.
        border = np.concatenate(
            [arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0
        )
        background_color = np.median(border, axis=0)
        background_distance = np.linalg.norm(arr - background_color, axis=2)
        foreground = background_distance > 24.0

        garment_mask = ~skin & foreground
        garment_mask = _clean_mask(garment_mask)

        key = clothing_type.lower()
        vertical_filter = np.ones((h, w), dtype=bool)
        label = "clothing item"
        if any(name in key for name in ("shirt", "t-shirt", "jacket", "blazer")):
            vertical_filter[int(h * 0.76):, :] = False
            label = "upper-clothes"
        elif "trousers" in key:
            vertical_filter[:int(h * 0.34), :] = False
            label = "pants"
        elif "dress" in key:
            label = "dress"
        garment_mask &= vertical_filter
        garment_mask = _clean_mask(garment_mask)

        return {
            "garment_mask": garment_mask,
            "labels":       [label],
            "label_mask":   garment_mask.astype(int),
            "confidence":   0.35,
        }

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _target_ids(clothing_type: str) -> list[int]:
        mapping = {
            "shirt / blouse":   [4],
            "t-shirt":          [4],
            "trousers":         [6],
            "dress":            [7],
            "jacket / blazer":  [4],
            "traditional wear": [4, 6, 7],
            "saree / kurta":    [4, 6, 7],
        }
        key = clothing_type.lower()
        for k, v in mapping.items():
            if k in key:
                return v
        # auto-detect: return only true garment labels (no body parts)
        return ALL_GARMENT_IDS


# ── utility ───────────────────────────────────────────────────────────────────
def _clean_mask(mask: np.ndarray, min_area: int | None = None) -> np.ndarray:
    """Close small gaps, fill holes, and remove tiny connected components."""
    min_area = min_area or max(64, int(mask.size * 0.0008))
    mask_u8 = mask.astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)

    try:
        from scipy import ndimage
        mask_u8 = ndimage.binary_fill_holes(mask_u8).astype(np.uint8)
        labeled, n = ndimage.label(mask_u8)
        sizes = ndimage.sum(mask_u8, labeled, range(1, n + 1))
        clean = np.zeros_like(mask_u8, dtype=bool)
        for i, s in enumerate(sizes, start=1):
            if s >= min_area:
                clean |= labeled == i
        return clean
    except ImportError:
        return mask_u8.astype(bool)
