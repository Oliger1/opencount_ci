# src/opencount_ci/detectors/log.py
from __future__ import annotations
from typing import List, Iterable, Tuple
import numpy as np

from ..core.base import BaseDetector
from ..core.geometry import non_max_suppression, blob_to_box

BBox = Tuple[int, int, int, int]
Blob = Tuple[int, int, float]


class LoGDetector(BaseDetector):
    """
    Multi-scale Laplacian-of-Gaussian (LoG) blob detector.

    Optimized with:
    - Multiple scales (log_sigmas)
    - Scale-normalized response (sigma^2 * |Laplacian|)
    - 3x3 dilation peak detection
    - Top-K pruning before NMS
    - Minimum area filtering
    """

    def __init__(
            self,
            log_sigmas=(2.0, 3.0, 4.0, 6.0, 8.0),
            log_threshold: float = 0.04,
            blob_margin_factor: float = 0.5,
            iou_threshold: float = 0.5,
            min_area: int = 64,
            max_blobs: int = 2000,
    ):
        super().__init__()
        self.log_sigmas = tuple(float(s) for s in log_sigmas)
        self.log_threshold = float(log_threshold)
        self.blob_margin_factor = float(blob_margin_factor)
        self.iou_threshold = float(iou_threshold)
        self.min_area = int(min_area)
        self.max_blobs = int(max_blobs)

    def detect(self, image: np.ndarray) -> List[BBox]:
        """Detect objects and return bounding boxes."""
        self.validate_image(image)

        blobs, scores = self._detect_blobs(
            image, self.log_sigmas, self.log_threshold
        )

        # Top-K pruning
        if len(blobs) > self.max_blobs:
            idx = np.argpartition(scores, -self.max_blobs)[-self.max_blobs:]
            blobs = [blobs[i] for i in idx]

        # Convert blobs to boxes
        boxes: List[BBox] = [
            blob_to_box(b, self.blob_margin_factor) for b in blobs
        ]

        # Filter by minimum area
        boxes = [b for b in boxes if self._box_area(b) >= self.min_area]

        # NMS
        boxes = non_max_suppression(boxes, self.iou_threshold)

        return boxes

    def _detect_blobs(
            self, gray: np.ndarray, sigmas: Iterable[float], threshold: float
    ) -> Tuple[List[Blob], np.ndarray]:
        """Detect blobs at multiple scales."""
        cv2 = self.cv2
        g = gray.astype(np.float32) / 255.0
        scales = list(sigmas)

        # Compute LoG response at each scale
        responses = [self._log_response(g, s) for s in scales]
        stack = np.stack(responses, axis=-1)  # H x W x S

        m = float(stack.max())
        if m <= 0.0:
            return [], np.array([], dtype=np.float32)

        norm = stack / m  # Normalize to [0,1]

        # Peak detection (2D dilation at each scale)
        peaks = np.zeros_like(norm, dtype=bool)
        kernel = np.ones((3, 3), np.uint8)

        for k in range(len(scales)):
            r = norm[:, :, k]
            mx = cv2.dilate(r, kernel)
            peaks[:, :, k] = (r == mx) & (r >= threshold)

        ys, xs, ks = np.where(peaks)

        if len(ys) == 0:
            return [], np.array([], dtype=np.float32)

        scores = norm[ys, xs, ks].astype(np.float32)
        blobs: List[Blob] = []

        for y, x, k in zip(ys, xs, ks):
            sigma = float(scales[k])
            radius = float(np.sqrt(2.0) * sigma)
            blobs.append((int(x), int(y), radius))

        return blobs, scores

    def _log_response(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """Compute scale-normalized LoG response: sigma^2 * |Laplacian|."""
        cv2 = self.cv2

        # Kernel size (odd) ~= 6*sigma + 1, capped at 31
        ksize = max(3, int(6 * sigma + 1) | 1)
        ksize = min(ksize, 31)

        blur = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        lap = cv2.Laplacian(blur, ddepth=cv2.CV_32F, ksize=3)

        return (sigma ** 2) * np.abs(lap)

    @staticmethod
    def _box_area(b: BBox) -> int:
        """Calculate box area."""
        x1, y1, x2, y2 = b
        return max(0, x2 - x1) * max(0, y2 - y1)