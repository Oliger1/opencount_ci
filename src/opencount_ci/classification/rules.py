# src/opencount_ci/classification/rules.py
"""
Base classes for anonymous rule-based classification.
No semantic labels - only visual feature detection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2

BBox = Tuple[int, int, int, int]


@dataclass
class Rule:
    """
    Base rule for visual pattern detection.
    Returns (pattern_id, confidence) or None.
    """
    name: str
    priority: int = 5

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        raise NotImplementedError


def _create_mask(gray: np.ndarray, threshold: int = 0) -> np.ndarray:
    """Create binary mask from grayscale."""
    if threshold == 0:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def _edge_density(gray: np.ndarray) -> float:
    """Calculate edge density."""
    edges = cv2.Canny(gray, 60, 150)
    return float(np.count_nonzero(edges)) / (gray.size + 1e-9)


class ShapeDetector:
    """Detects geometric shape patterns."""

    @staticmethod
    def circularity(mask: np.ndarray) -> float:
        """Measure circularity (0-1)."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True) + 1e-6
        return float(4.0 * np.pi * (area / (perimeter * perimeter)))

    @staticmethod
    def rectangularity(mask: np.ndarray) -> float:
        """Measure rectangularity (0-1)."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = w * h
        return float(area / (bbox_area + 1e-9))

    @staticmethod
    def classify_shape(mask: np.ndarray) -> Tuple[str, float]:
        """
        Classify shape into generic categories.
        Returns: (shape_type, confidence)
        Types: 'circular', 'rectangular', 'elliptical', 'irregular'
        """
        circ = ShapeDetector.circularity(mask)
        rect = ShapeDetector.rectangularity(mask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return ('irregular', 0.0)

        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        aspect = float(w) / float(h + 1e-6)

        if circ > 0.75:
            return ('circular', circ)
        elif rect > 0.8:
            return ('rectangular', rect)
        elif 0.5 < circ < 0.75 and (aspect > 1.5 or aspect < 0.67):
            return ('elliptical', circ)
        else:
            return ('irregular', 1.0 - max(circ, rect))


class ColorDetector:
    """Detects color patterns without semantic naming."""

    @staticmethod
    def dominant_hue_range(hsv: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[int, float]:
        """
        Get dominant hue range (0-7 for 8 sectors of hue circle).
        Returns: (hue_sector, confidence)
        """
        if mask is not None:
            pixels = hsv[mask > 0]
        else:
            pixels = hsv.reshape(-1, 3)

        if len(pixels) == 0:
            return (0, 0.0)

        # Filter by saturation (ignore grays)
        saturated = pixels[pixels[:, 1] > 30]
        if len(saturated) == 0:
            return (0, 0.0)

        hues = saturated[:, 0]

        # Divide hue circle into 8 sectors (0-7)
        # Each sector is 22.5 degrees (180/8 for OpenCV hue range)
        sector = int(np.median(hues) / 22.5)

        # Confidence based on hue consistency
        hue_std = float(np.std(hues))
        confidence = float(1.0 / (1.0 + hue_std / 20.0))

        return (sector, confidence)

    @staticmethod
    def saturation_level(hsv: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Classify saturation: 'low', 'medium', 'high'
        Returns: (level, confidence)
        """
        if mask is not None:
            pixels = hsv[mask > 0]
        else:
            pixels = hsv.reshape(-1, 3)

        if len(pixels) == 0:
            return ('low', 0.0)

        sat_mean = float(np.mean(pixels[:, 1]))
        sat_std = float(np.std(pixels[:, 1]))

        confidence = 1.0 / (1.0 + sat_std / 30.0)

        if sat_mean < 50:
            return ('low', confidence)
        elif sat_mean < 120:
            return ('medium', confidence)
        else:
            return ('high', confidence)

    @staticmethod
    def value_level(hsv: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Classify brightness: 'dark', 'medium', 'bright'
        Returns: (level, confidence)
        """
        if mask is not None:
            pixels = hsv[mask > 0]
        else:
            pixels = hsv.reshape(-1, 3)

        if len(pixels) == 0:
            return ('medium', 0.0)

        val_mean = float(np.mean(pixels[:, 2]))
        val_std = float(np.std(pixels[:, 2]))

        confidence = 1.0 / (1.0 + val_std / 30.0)

        if val_mean < 70:
            return ('dark', confidence)
        elif val_mean < 170:
            return ('medium', confidence)
        else:
            return ('bright', confidence)

    @staticmethod
    def dominant_color_name(hsv: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Map to basic color names for compatibility.
        Returns generic color category without specific object association.
        """
        hue_sector, hue_conf = ColorDetector.dominant_hue_range(hsv, mask)
        sat_level, sat_conf = ColorDetector.saturation_level(hsv, mask)

        if sat_level == 'low':
            return ('gray', sat_conf)

        # Map hue sectors to color names (generic, not object-specific)
        color_map = {
            0: 'red',
            1: 'orange',
            2: 'yellow',
            3: 'green',
            4: 'cyan',
            5: 'blue',
            6: 'purple',
            7: 'red'
        }

        color = color_map.get(hue_sector, 'unknown')
        confidence = (hue_conf + sat_conf) / 2.0

        return (color, confidence)


class TextureDetector:
    """Detects texture patterns."""

    @staticmethod
    def texture_type(gray: np.ndarray) -> Tuple[str, float]:
        """
        Classify texture: 'smooth', 'textured', 'structured'
        Returns: (type, confidence)
        """
        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_dens = float(np.count_nonzero(edges)) / (gray.size + 1e-9)

        # Texture variance
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture_var = float(lap.var())

        if texture_var < 50:
            return ('smooth', 1.0 / (1.0 + texture_var / 20.0))
        elif edge_dens > 0.1:
            return ('structured', min(1.0, edge_dens / 0.15))
        else:
            return ('textured', min(1.0, texture_var / 200.0))


class GenericRuleClassifier:
    """
    Generic classifier using visual rules without semantic labels.
    """

    def __init__(self):
        self.shape_detector = ShapeDetector()
        self.color_detector = ColorDetector()
        self.texture_detector = TextureDetector()

    def classify(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Classify into generic visual category.
        Returns: (category, confidence)
        Category format: "shape_texture_color" e.g., "circular_smooth_red"
        """
        if crop_bgr.size == 0:
            return ('unknown', 0.0)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = _create_mask(gray)

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # Analyze features
        shape, shape_conf = self.shape_detector.classify_shape(mask)
        texture, texture_conf = self.texture_detector.texture_type(gray)
        color, color_conf = self.color_detector.dominant_color_name(hsv, mask)

        # Build category label
        category = f"{shape}_{texture}_{color}"
        confidence = (shape_conf + texture_conf + color_conf) / 3.0

        return (category, float(confidence))