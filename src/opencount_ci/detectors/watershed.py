# src/opencount_ci/detectors/watershed.py
from __future__ import annotations
from typing import List
import numpy as np

from ..core.base import BaseDetector, BBox


class WatershedDetector(BaseDetector):
    """Marker-controlled Watershed detector."""

    def __init__(
            self,
            min_area=64,
            peak_rel=0.35,
            adaptive_kernel=True,
            morph_kernel_size=3,
            morph_iterations=1,
            dilation_iterations=2
    ):
        super().__init__()
        self.min_area = int(min_area)
        self.peak_rel = float(peak_rel)
        self.adaptive_kernel = bool(adaptive_kernel)
        self.morph_kernel_size = int(morph_kernel_size)
        self.morph_iterations = int(morph_iterations)
        self.dilation_iterations = int(dilation_iterations)

    def detect(self, image: np.ndarray) -> List[BBox]:
        """Detect objects using watershed segmentation."""
        self.validate_image(image)
        cv2 = self.cv2

        # Threshold
        _, th = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological operations
        if self.adaptive_kernel:
            size = max(3, min(7, max(image.shape) // 200))
        else:
            size = self.morph_kernel_size

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        opened = cv2.morphologyEx(
            th, cv2.MORPH_OPEN, k, iterations=self.morph_iterations
        )
        sure_bg = cv2.dilate(opened, k, iterations=self.dilation_iterations)

        # Distance transform
        dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

        peak_thr = max(0.2, float(self.peak_rel))
        _, sure_fg = cv2.threshold(
            (dist_norm * 255).astype(np.uint8),
            int(255 * peak_thr),
            255,
            cv2.THRESH_BINARY
        )

        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        n_labels, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
        markers = markers + 1
        markers[unknown == 255] = 0

        # Watershed
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.watershed(color_img, markers)

        # Extract boxes
        boxes: List[BBox] = []
        for label in range(2, n_labels + 1):
            mask = (markers == label).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not cnts:
                continue

            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))

            if w * h >= self.min_area:
                boxes.append((int(x), int(y), int(x + w), int(y + h)))

        return boxes