import unittest

import cv2
import numpy as np
from PIL import Image

from models.ironing_pipeline import IroningPipeline
from models.segmentation import _clean_mask
from models.wrinkle_detector import WrinkleDetector


def _synthetic_garment():
    arr = np.full((180, 180, 3), 245, dtype=np.uint8)
    mask = np.zeros((180, 180), dtype=bool)
    mask[25:165, 30:150] = True
    arr[mask] = (105, 145, 195)

    # A colored graphic should not be interpreted as wrinkle shading.
    arr[55:85, 70:110] = (190, 55, 60)

    # Dark and bright curved crease pairs.
    xs = np.arange(40, 140)
    for y in (105, 122, 139):
        wave = (y + 4 * np.sin(xs / 8.0)).astype(np.int32)
        points = np.column_stack([xs, wave])
        cv2.polylines(arr, [points], False, (55, 90, 135), 2, cv2.LINE_AA)
        points[:, 1] += 3
        cv2.polylines(arr, [points], False, (155, 190, 230), 1, cv2.LINE_AA)

    return Image.fromarray(arr), mask


def _synthetic_broad_folds():
    h, w = 220, 300
    mask = np.zeros((h, w), dtype=bool)
    mask[20:205, 20:280] = True
    base = np.array([160, 190, 220], dtype=np.float32)
    arr = np.full((h, w, 3), 245, dtype=np.float32)
    arr[mask] = base

    y, x = np.indices((h, w))
    shading = (
        -32 * np.exp(-((x - 92) / 17) ** 2)
        + 20 * np.exp(-((x - 128) / 19) ** 2)
        - 27 * np.exp(-((x - 205) / 22) ** 2)
        + 8 * np.sin(y / 16.0)
    )
    arr[mask] += shading[mask, np.newaxis]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)), mask


class IroningPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = WrinkleDetector(device="cpu", use_cnn=False)
        cls.pipeline = IroningPipeline(device="cpu")

    def test_ironing_reduces_wrinkles_and_preserves_background(self):
        image, mask = _synthetic_garment()
        before = np.array(image)
        wrinkle_map = self.detector.detect(image, mask)["wrinkle_map"]

        result = self.pipeline.iron(
            image, mask, wrinkle_map, intensity=0.8,
            fabric_type="cotton", preserve_structural_folds=False,
        )
        after = np.array(result)

        self.assertTrue(np.array_equal(before[~mask], after[~mask]))

        gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hp_before = np.abs(gray_before - cv2.GaussianBlur(gray_before, (0, 0), 2))
        hp_after = np.abs(gray_after - cv2.GaussianBlur(gray_after, (0, 0), 2))
        crease_zone = np.zeros_like(mask)
        crease_zone[95:150, 35:145] = True

        contrast_ratio = hp_after[crease_zone].mean() / hp_before[crease_zone].mean()
        brightness_shift = abs((gray_after - gray_before)[mask].mean())
        self.assertLess(contrast_ratio, 0.72)
        self.assertLess(brightness_shift, 3.0)

    def test_colored_graphic_is_preserved(self):
        image, mask = _synthetic_garment()
        before = np.array(image)
        wrinkle_map = self.detector.detect(image, mask)["wrinkle_map"]
        after = np.array(
            self.pipeline.iron(
                image, mask, wrinkle_map, intensity=1.0,
                fabric_type="cotton", preserve_structural_folds=False,
            )
        )

        logo_delta = np.abs(
            after[58:82, 73:107].astype(np.int16)
            - before[58:82, 73:107].astype(np.int16)
        ).mean()
        self.assertLess(logo_delta, 2.0)

    def test_empty_mask_returns_unchanged_image(self):
        image, _ = _synthetic_garment()
        empty = np.zeros((180, 180), dtype=bool)
        result = self.pipeline.iron(image, empty, empty.astype(np.float32))
        self.assertTrue(np.array_equal(np.array(image), np.array(result)))

    def test_medium_press_reduces_broad_fold_shading(self):
        image, mask = _synthetic_broad_folds()
        before = np.array(image)
        wrinkle_map = self.detector.detect(image, mask)["wrinkle_map"]
        after = np.array(
            self.pipeline.iron(
                image, mask, wrinkle_map, intensity=0.8,
                fabric_type="cotton", preserve_structural_folds=False,
            )
        )

        gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
        broad_before = np.abs(
            cv2.GaussianBlur(gray_before, (0, 0), 6)
            - cv2.GaussianBlur(gray_before, (0, 0), 32)
        )
        broad_after = np.abs(
            cv2.GaussianBlur(gray_after, (0, 0), 6)
            - cv2.GaussianBlur(gray_after, (0, 0), 32)
        )
        interior = np.zeros_like(mask)
        interior[45:185, 45:255] = True
        self.assertLess(
            broad_after[interior].mean() / broad_before[interior].mean(), 0.72
        )

    def test_mask_cleanup_fills_holes_and_removes_tiny_islands(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        mask[40:50, 40:50] = False
        mask[2:4, 2:4] = True

        cleaned = _clean_mask(mask, min_area=100)
        self.assertTrue(cleaned[45, 45])
        self.assertFalse(cleaned[2, 2])


if __name__ == "__main__":
    unittest.main()
