"""
Ironing Pipeline
Combines:
  1. Classical wrinkle suppression (bilateral filter + guided smoothing)
  2. Stable Diffusion inpainting for realistic texture reconstruction
  3. Fabric-specific post-processing
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import torch


# Fabric-specific ironing parameters
FABRIC_PARAMS = {
    "cotton":         {"smooth_sigma": 3.0, "texture_strength": 0.85, "sharpness": 1.0},
    "linen":          {"smooth_sigma": 2.5, "texture_strength": 0.90, "sharpness": 1.1},
    "silk":           {"smooth_sigma": 4.5, "texture_strength": 0.70, "sharpness": 0.7},
    "denim":          {"smooth_sigma": 2.0, "texture_strength": 0.95, "sharpness": 1.3},
    "polyester":      {"smooth_sigma": 3.5, "texture_strength": 0.75, "sharpness": 0.9},
    "wool":           {"smooth_sigma": 3.0, "texture_strength": 0.80, "sharpness": 0.8},
    "synthetic blend":{"smooth_sigma": 3.2, "texture_strength": 0.78, "sharpness": 0.95},
    "auto-detect":    {"smooth_sigma": 3.0, "texture_strength": 0.82, "sharpness": 1.0},
}

SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"


class IroningPipeline:
    """
    Main ironing engine.
    Uses a deterministic, identity-preserving classical approach by default.
    Stable Diffusion can be explicitly enabled, but is kept off by default
    because unconstrained inpainting can change logos, garment shape, or color.
    """

    def __init__(self, device: str | None = None, enable_diffusion: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_diffusion = enable_diffusion
        self._pipe  = None
        if self.enable_diffusion and self.device == "cuda":
            self._load_diffusion()

    # ── model loading ─────────────────────────────────────────────────────────
    def _load_diffusion(self):
        try:
            from diffusers import StableDiffusionInpaintPipeline
            print(f"[IroningPipeline] Loading SD inpainting on {self.device} …")
            self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                SD_MODEL_ID,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(self.device)
            self._pipe.enable_attention_slicing()
            print("[IroningPipeline] Diffusion pipeline ready.")
        except Exception as exc:
            print(f"[IroningPipeline] Could not load diffusion model: {exc}")
            print("[IroningPipeline] Falling back to classical ironing.")
            self._pipe = None

    # ── public API ────────────────────────────────────────────────────────────
    def iron(
        self,
        image: Image.Image,
        garment_mask: np.ndarray,
        wrinkle_map: np.ndarray,
        intensity: float = 0.7,
        fabric_type: str = "auto-detect",
        preserve_structural_folds: bool = True,
    ) -> Image.Image:
        """
        Parameters
        ----------
        image                    : PIL RGB image (already resized)
        garment_mask             : H×W bool – True = garment pixel
        wrinkle_map              : H×W float32 – wrinkle probability
        intensity                : 0.0 (none) → 1.0 (max)
        fabric_type              : key into FABRIC_PARAMS
        preserve_structural_folds: if True, deep folds near joints are kept

        Returns
        -------
        Ironed PIL RGB image
        """
        params = FABRIC_PARAMS.get(fabric_type.lower(), FABRIC_PARAMS["auto-detect"])

        garment_mask = garment_mask.astype(bool)
        if not garment_mask.any():
            return image.copy()

        treatment_mask = self._build_treatment_mask(
            garment_mask=garment_mask,
            wrinkle_map=wrinkle_map,
            intensity=intensity,
        )

        if preserve_structural_folds:
            treatment_mask = self._exclude_structural_folds(
                treatment_mask, wrinkle_map, garment_mask
            )

        # Never let blur or inpainting touch a non-garment pixel.
        treatment_mask *= garment_mask.astype(np.float32)

        if self._pipe is not None and intensity >= 0.5:
            ironed = self._diffusion_iron(
                image, garment_mask, treatment_mask, intensity, fabric_type, params
            )
        else:
            ironed = self._classical_iron(
                image, garment_mask, treatment_mask, intensity, params
            )

        ironed = self._postprocess(
            ironed, garment_mask, treatment_mask, params, intensity
        )

        return ironed

    @staticmethod
    def _build_treatment_mask(
        garment_mask: np.ndarray,
        wrinkle_map: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Create a soft mask focused on likely wrinkles, away from boundaries."""
        garment_f = garment_mask.astype(np.float32)
        wmap = np.nan_to_num(wrinkle_map.astype(np.float32), nan=0.0)
        values = wmap[garment_mask]

        if values.size:
            low, high = np.percentile(values, (35, 98))
            if high > low + 1e-6:
                wmap = np.clip((wmap - low) / (high - low), 0.0, 1.0)
            else:
                wmap = np.zeros_like(wmap)

        # Keep detected wrinkle zones stronger, but apply a meaningful base
        # press across the garment. Real photographs often contain wide,
        # low-contrast folds that local ridge detectors score too weakly.
        wmap = np.sqrt(np.clip(wmap, 0.0, 1.0))
        wmap = cv2.GaussianBlur(wmap, (0, 0), 3.0)
        coverage_floor = 0.18 + 0.55 * float(np.clip(intensity, 0.0, 1.0))
        treatment = garment_f * np.clip(
            coverage_floor + (1.0 - coverage_floor) * wmap, 0.0, 1.0
        )

        # Feather inward at the garment boundary so skin/background remain exact.
        dist = cv2.distanceTransform((garment_mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)
        edge_width = max(3.0, min(12.0, min(garment_mask.shape) * 0.015))
        inward_feather = np.clip(dist / edge_width, 0.0, 1.0)
        return (treatment * inward_feather).astype(np.float32)

    # ── diffusion path ────────────────────────────────────────────────────────
    def _diffusion_iron(
        self,
        image: Image.Image,
        garment_mask: np.ndarray,
        treatment_mask: np.ndarray,
        intensity: float,
        fabric_type: str,
        params: dict,
    ) -> Image.Image:
        h, w = image.size[1], image.size[0]

        # SD works best at 512×512 or 768×768
        sd_size = 768 if max(h, w) > 600 else 512
        img_sd  = image.resize((sd_size, sd_size), Image.LANCZOS)

        # inpainting mask: white = areas to regenerate (wrinkled regions)
        mask_np  = ((treatment_mask > 0.45).astype(np.uint8) * 255)
        mask_np  = cv2.resize(mask_np, (sd_size, sd_size), interpolation=cv2.INTER_NEAREST)
        # dilate slightly so SD gets context pixels
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_np  = cv2.dilate(mask_np, kernel)
        mask_pil = Image.fromarray(mask_np)

        prompt = (
            f"a person wearing freshly ironed {fabric_type} clothing, "
            "smooth wrinkle-free fabric, sharp crisp appearance, "
            "natural lighting, photorealistic, high quality, 4k"
        )
        neg_prompt = (
            "wrinkles, creases, folds, crumpled, messy, distorted face, "
            "changed clothing color, artifacts, blurry"
        )

        result = self._pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=img_sd,
            mask_image=mask_pil,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=min(0.35 + intensity * 0.25, 0.75),
        ).images[0]

        # upscale back to original size
        result = result.resize((w, h), Image.LANCZOS)

        # composite: only replace treated pixels
        arr_orig   = np.array(image).astype(float)
        arr_ironed = np.array(result).astype(float)
        alpha      = cv2.resize(treatment_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha      = alpha[..., np.newaxis] * intensity

        blended = (arr_ironed * alpha + arr_orig * (1 - alpha)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    # ── classical path ────────────────────────────────────────────────────────
    def _classical_iron(
        self,
        image: Image.Image,
        garment_mask: np.ndarray,
        treatment_mask: np.ndarray,
        intensity: float,
        params: dict,
    ) -> Image.Image:
        """Suppress wrinkle shading while preserving hue and meaningful edges."""
        arr  = np.array(image).astype(np.float32)
        lab = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

        # Separate luminance into wrinkle-sized frequency bands, then compress
        # those bands. Normalized blurs only borrow color from garment pixels,
        # preventing skin/background halos near the garment boundary.
        g2 = _normalized_blur(L, garment_mask, 2.0)
        g8 = _normalized_blur(L, garment_mask, 8.0)
        g28 = _normalized_blur(L, garment_mask, 28.0)
        g55 = _normalized_blur(L, garment_mask, 55.0)

        fine_factor = 0.96 - 0.42 * intensity
        medium_factor = 0.62 - 0.58 * intensity
        broad_factor = 0.64 - 0.54 * intensity
        lighting_factor = 0.92 - 0.30 * intensity
        target = (
            g55
            + fine_factor * (L - g2)
            + medium_factor * (g2 - g8)
            + broad_factor * (g8 - g28)
            + lighting_factor * (g28 - g55)
        )

        # Avoid a global brightness shift after flattening.
        target += np.median(L[garment_mask]) - np.median(target[garment_mask])

        # Protect colored graphics and the strongest structural edges. Wrinkle
        # shading is usually luminance-only; logos commonly have chroma edges.
        chroma_grad = np.hypot(_gradient_magnitude(A), _gradient_magnitude(B))
        lum_grad = _gradient_magnitude(L)
        chroma_protect = np.clip((chroma_grad - 8.0) / 28.0, 0.0, 1.0)
        local_a = _normalized_blur(A, garment_mask, 18.0)
        local_b = _normalized_blur(B, garment_mask, 18.0)
        color_region_delta = np.hypot(A - local_a, B - local_b)
        color_region_protect = np.clip(
            (color_region_delta - 5.0) / 12.0, 0.0, 1.0
        )
        lum_protect = _robust_unit_scale(lum_grad, garment_mask)
        hard_edge_protect = np.clip((lum_protect - 0.97) / 0.03, 0.0, 1.0)
        edge_protect = np.maximum.reduce(
            [chroma_protect, color_region_protect, hard_edge_protect]
        )

        press_strength = 0.45 + 0.55 * intensity
        alpha = treatment_mask * press_strength * (1.0 - 0.985 * edge_protect)
        alpha = np.clip(alpha, 0.0, 0.95)
        L_ironed = np.clip(target * alpha + L * (1.0 - alpha), 0, 255)

        lab_ironed = lab.copy()
        lab_ironed[..., 0] = L_ironed
        result = cv2.cvtColor(lab_ironed.astype(np.uint8), cv2.COLOR_LAB2RGB)

        # LAB round-tripping can move untouched pixels by a value or two.
        result[~garment_mask] = arr.astype(np.uint8)[~garment_mask]
        return Image.fromarray(result)

    # ── structural fold preservation ──────────────────────────────────────────
    @staticmethod
    def _exclude_structural_folds(
        treatment_mask: np.ndarray,
        wrinkle_map: np.ndarray,
        garment_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Very strong wrinkles near garment boundaries are likely structural
        (arm bends, waist line). Reduce their treatment weight.
        """
        h, w = garment_mask.shape

        # distance transform from garment edges
        garment_u8 = garment_mask.astype(np.uint8) * 255
        dist       = cv2.distanceTransform(garment_u8, cv2.DIST_L2, 5)
        dist      /= (dist.max() + 1e-6)

        # near boundary (dist < 0.1) AND very strong wrinkle → structural
        structural_prob = (1.0 - dist) * (wrinkle_map > 0.6).astype(float)
        structural_prob = cv2.GaussianBlur(structural_prob.astype(np.float32), (15, 15), 4.0)

        # Attenuate rather than fully preserve them. A virtual press should
        # still visibly soften folds near elbows and shoulders.
        treatment_mask = treatment_mask * (1.0 - structural_prob * 0.35)
        return treatment_mask.astype(np.float32)

    # ── topology flattening ────────────────────────────────────────────────────
    @staticmethod
    def _flatten_topology(
        original: np.ndarray,
        smoothed: np.ndarray,
        treatment_mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """
        Re-distribute luminance to reduce the shadow/highlight gradient
        that wrinkles create, without altering hue.
        """
        orig_hsv   = cv2.cvtColor(original.astype(np.uint8),  cv2.COLOR_RGB2HSV).astype(float)
        smooth_hsv = cv2.cvtColor(smoothed.astype(np.uint8),  cv2.COLOR_RGB2HSV).astype(float)

        # only flatten luminance channel (V) — both are 2D (H, W), so use the mask directly
        weight = treatment_mask * intensity
        smooth_hsv[..., 2] = (
            smooth_hsv[..., 2] * weight +
            orig_hsv[..., 2]   * (1.0 - weight)
        )

        flat = cv2.cvtColor(smooth_hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return flat.astype(np.float32)

    # ── fabric post-processing ────────────────────────────────────────────────
    @staticmethod
    def _postprocess(
        image: Image.Image,
        garment_mask: np.ndarray,
        treatment_mask: np.ndarray,
        params: dict,
        intensity: float,
    ) -> Image.Image:
        # The previous kernel had a sum below 1.0, visibly darkening garments
        # and sharpening wrinkles. Edge protection in _classical_iron already
        # preserves useful detail, so no global sharpening is needed.
        return image


def _gradient_magnitude(channel: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _normalized_blur(
    channel: np.ndarray,
    garment_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    mask = garment_mask.astype(np.float32)
    numerator = cv2.GaussianBlur(channel * mask, (0, 0), sigma)
    denominator = cv2.GaussianBlur(mask, (0, 0), sigma)
    blurred = numerator / np.maximum(denominator, 0.03)
    blurred[~garment_mask] = channel[~garment_mask]
    return blurred.astype(np.float32)


def _robust_unit_scale(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    if not selected.size:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(selected, (60, 98.5))
    if high <= low + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


# ── guided filter helper ──────────────────────────────────────────────────────
def _guided_filter(
    src: np.ndarray,
    guide: np.ndarray,
    radius: int = 15,
    eps: float = 0.01,
) -> np.ndarray:
    """Pure-NumPy approximation of the guided image filter."""
    src   = src   / 255.0
    guide = guide / 255.0
    r = max(radius, 1)

    def box(img, r):
        from numpy.lib.stride_tricks import sliding_window_view
        h, w = img.shape[:2]
        out  = np.zeros_like(img)
        kh = min(r * 2 + 1, h)
        kw = min(r * 2 + 1, w)
        # simple uniform filter
        import scipy.ndimage as nd
        for c in range(img.shape[2] if img.ndim == 3 else 1):
            sl = img[..., c] if img.ndim == 3 else img
            out[..., c] = nd.uniform_filter(sl, size=(kh, kw)) if img.ndim == 3 else nd.uniform_filter(sl, size=(kh, kw))
        return out

    mean_I  = box(guide, r)
    mean_p  = box(src,   r)
    mean_Ip = box(guide * src, r)
    cov_Ip  = mean_Ip - mean_I * mean_p

    mean_II = box(guide * guide, r)
    var_I   = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box(a, r)
    mean_b = box(b, r)

    out = (mean_a * guide + mean_b).clip(0, 1) * 255.0
    return out.astype(np.float32)
