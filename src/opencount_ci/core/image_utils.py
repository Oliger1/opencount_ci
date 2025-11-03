# src/opencount_ci/core/image_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from ..exceptions import ImageLoadError


class ImageProcessor:
    """Load, enhance, and perturb grayscale images."""

    def __init__(self):
        self._cv2 = None

    @property
    def cv2(self):
        """Lazy import of OpenCV."""
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2

    def load_image(self, path: str, max_size: Optional[int] = None) -> np.ndarray:
        """Load and optionally resize image."""
        p = Path(path)

        if not p.exists():
            raise ImageLoadError(f"Image not found: {path}")

        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            raise ImageLoadError(f"Unsupported image extension: {p.suffix}")

        img = self.cv2.imread(str(p), self.cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ImageLoadError(f"Failed to read image: {path}")

        if max_size and max(img.shape) > max_size:
            img = self._resize(img, max_size)

        return img

    def _resize(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """Resize image maintaining aspect ratio."""
        h, w = image.shape
        if max(h, w) <= max_size:
            return image

        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return self.cv2.resize(image, (new_w, new_h), interpolation=self.cv2.INTER_AREA)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement."""
        clahe = self.cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def perturb(
            self,
            image: np.ndarray,
            seed: int,
            noise_std_range=(0.01, 0.03),
            gamma_range=(0.85, 1.15),
            brightness_range=(0.95, 1.05),
            blur_probability=0.3
    ) -> np.ndarray:
        """Apply random perturbations for bootstrap."""
        rng = np.random.default_rng(seed)
        g = image.astype(np.float32) / 255.0

        # Add noise
        noise_std = float(rng.uniform(*noise_std_range))
        noise = rng.normal(0, noise_std, g.shape).astype(np.float32)
        g = np.clip(g + noise, 0.0, 1.0)

        # Gamma correction
        gamma = float(rng.uniform(*gamma_range))
        g = np.clip(g ** gamma, 0.0, 1.0)

        # Brightness adjustment
        bright = float(rng.uniform(*brightness_range))
        g = np.clip(g * bright, 0.0, 1.0)

        # Optional blur
        if rng.random() < blur_probability:
            k = int(rng.choice([3, 5]))
            g = self.cv2.GaussianBlur(g, (k, k), 0)

        return (g * 255.0).astype(np.uint8)