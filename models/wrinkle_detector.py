"""
Wrinkle Detection Module
Detects wrinkles, creases, and fabric deformation using multi-scale
edge analysis + a lightweight CNN classifier.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Lightweight wrinkle-scoring CNN ──────────────────────────────────────────
class WrinkleCNN(nn.Module):
    """
    Small encoder that produces a per-pixel wrinkle probability map.
    Input : 3 × H × W (RGB, normalised)
    Output: 1 × H × W (probability 0-1)
    """

    def __init__(self):
        super().__init__()
        self.enc1 = self._block(3,   32)
        self.enc2 = self._block(32,  64)
        self.enc3 = self._block(64, 128)
        self.dec2 = self._block(128 + 64, 64)
        self.dec1 = self._block(64  + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))

        d2 = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.head(d1))


# ── Main detector class ───────────────────────────────────────────────────────
class WrinkleDetector:
    """
    Combines classical edge/texture cues with optional CNN refinement.
    """

    WRINKLE_MODEL_HF = os.getenv("WRINKLE_MODEL_HF")

    def __init__(self, device: str | None = None, use_cnn: bool = True):
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cnn = use_cnn
        self._cnn    = None
        if use_cnn:
            self._init_cnn()

    # ── init ──────────────────────────────────────────────────────────────────
    def _init_cnn(self):
        if not self.WRINKLE_MODEL_HF:
            print("[WrinkleDetector] No CNN repository configured; using classical detection.")
            self._cnn = None
            return

        self._cnn = WrinkleCNN().to(self.device).eval()
        # Attempt to load pre-trained weights; silently fall back to random init
        # (random-init will still work because we blend with classical cues)
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self.WRINKLE_MODEL_HF, "wrinkle_cnn.pth")
            state = torch.load(path, map_location=self.device)
            self._cnn.load_state_dict(state)
            print("[WrinkleDetector] Loaded CNN weights from HF Hub.")
        except Exception:
            print("[WrinkleDetector] No pre-trained CNN weights found; using classical detection only.")
            self._cnn = None

    # ── public ────────────────────────────────────────────────────────────────
    def detect(
        self,
        image: Image.Image,
        garment_mask: np.ndarray,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
          wrinkle_map   – H×W float32 [0,1], 1 = strong wrinkle
          wrinkle_score – scalar 0-100
          zones         – list of zone dicts {name, score, bbox}
        """
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

        # ── classical multi-scale wrinkle map ────────────────────────────────
        classical = self._classical_map(arr, garment_mask)

        # ── optional CNN refinement ───────────────────────────────────────────
        if self._cnn is not None:
            cnn_map = self._cnn_map(arr, garment_mask)
            wrinkle_map = 0.4 * classical + 0.6 * cnn_map
        else:
            wrinkle_map = classical

        # apply garment mask
        wrinkle_map = wrinkle_map * garment_mask.astype(float)

        # global score
        garment_pixels = garment_mask.sum()
        score = 0.0
        if garment_pixels > 0:
            score = float((wrinkle_map[garment_mask] > 0.3).sum()) / garment_pixels * 100.0
            score = min(score * 1.3, 100.0)  # scale to perceptible range

        # zone analysis
        zones = self._zone_analysis(wrinkle_map, garment_mask, h, w)

        return {
            "wrinkle_map":   wrinkle_map.astype(np.float32),
            "wrinkle_score": round(score, 1),
            "zones":         zones,
        }

    # ── classical detection ───────────────────────────────────────────────────
    @staticmethod
    def _classical_map(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros(mask.shape, dtype=np.float32)

        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        lum, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

        # Wrinkles appear as thin dark/bright ridges and local luminance bands.
        ridge = np.zeros(mask.shape, dtype=np.float32)
        lum_u8 = lum.astype(np.uint8)
        for size in (5, 9, 15):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            dark = cv2.morphologyEx(lum_u8, cv2.MORPH_BLACKHAT, kernel)
            bright = cv2.morphologyEx(lum_u8, cv2.MORPH_TOPHAT, kernel)
            ridge = np.maximum(ridge, np.maximum(dark, bright).astype(np.float32))

        # Difference-of-Gaussians adds broader wrinkle shading that morphology
        # alone can miss.
        fine = cv2.GaussianBlur(lum, (0, 0), 1.0)
        medium = cv2.GaussianBlur(lum, (0, 0), 5.0)
        broad = cv2.GaussianBlur(lum, (0, 0), 13.0)
        bands = 0.65 * np.abs(fine - medium) + 0.35 * np.abs(medium - broad)
        combined = 0.65 * ridge + 0.35 * bands

        # Colored boundaries are more likely logos/patterns than wrinkles.
        chroma_edge = np.hypot(_gradient_magnitude(a), _gradient_magnitude(b))
        chroma_scale = _robust_scale(chroma_edge, mask, 65, 98.5)
        combined *= 1.0 - 0.75 * chroma_scale

        # Exclude the garment contour, which was a major source of false
        # wrinkle detections in the previous Canny-based implementation.
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
        combined *= eroded.astype(np.float32)
        combined = _robust_scale(combined, eroded.astype(bool), 45, 99)
        return cv2.GaussianBlur(combined, (0, 0), 1.2).astype(np.float32)

    # ── CNN path ─────────────────────────────────────────────────────────────
    def _cnn_map(self, arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._cnn(tensor)                       # 1×1×H×W
        prob = out.squeeze().cpu().numpy()
        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return prob

    # ── zone analysis ─────────────────────────────────────────────────────────
    @staticmethod
    def _zone_analysis(
        wrinkle_map: np.ndarray,
        garment_mask: np.ndarray,
        h: int, w: int,
    ) -> list[dict]:
        zones_def = {
            "shoulder region": (0,      h // 4),
            "chest / torso":   (h // 4, h // 2),
            "mid-body":        (h // 2, 3 * h // 4),
            "lower body":      (3 * h // 4, h),
        }
        results = []
        for name, (y0, y1) in zones_def.items():
            zone_mask    = garment_mask[y0:y1, :]
            zone_wrinkle = wrinkle_map[y0:y1, :]
            n_px = zone_mask.sum()
            if n_px == 0:
                continue
            score = float(zone_wrinkle[zone_mask > 0].mean()) if n_px > 0 else 0.0
            results.append({
                "name":  name,
                "score": round(score * 100, 1),
                "bbox":  (0, y0, w, y1),
                "level": "high" if score > 0.45 else ("medium" if score > 0.25 else "low"),
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


def _gradient_magnitude(channel: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _robust_scale(
    values: np.ndarray,
    mask: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    selected = values[mask]
    if not selected.size:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(selected, (low_percentile, high_percentile))
    if high <= low + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
