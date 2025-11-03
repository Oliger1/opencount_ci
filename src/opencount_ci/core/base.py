# src/opencount_ci/core/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

BBox = Tuple[int, int, int, int]
Blob = Tuple[int, int, float]


class BaseDetector(ABC):
    """Abstract base class for all detectors."""

    def __init__(self):
        self._cv2 = None

    @property
    def cv2(self):
        """Lazy import of OpenCV."""
        if self._cv2 is None:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError as e:
                raise ImportError(
                    "OpenCV is required: pip install opencv-python"
                ) from e
        return self._cv2

    def validate_image(self, image: np.ndarray) -> None:
        """Validate input image format and type."""
        if image is None or image.size == 0:
            raise ValueError("Image is empty or None")
        if image.ndim != 2:
            raise ValueError("Image must be grayscale (2D) np.uint8")
        if image.dtype != np.uint8:
            raise ValueError(f"Image must be uint8, got {image.dtype}")

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[BBox]:
        """Detect objects and return bounding boxes."""
        ...