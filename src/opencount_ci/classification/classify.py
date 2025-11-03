# src/opencount_ci/classification/classify.py
"""
Anonymous classification module - groups objects by visual similarity
WITHOUT semantic labels (no 'apple', 'car', etc.)
"""
from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
import cv2


def _create_mask(gray: np.ndarray, threshold: int = 0) -> np.ndarray:
    """Create binary mask from grayscale crop."""
    if threshold == 0:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def _edge_density(gray: np.ndarray) -> float:
    """Calculate edge pixel density."""
    edges = cv2.Canny(gray, 60, 150)
    return float(np.count_nonzero(edges)) / (gray.shape[0] * gray.shape[1] + 1e-9)


class VisualFeatureExtractor:
    """Extract visual features without semantic interpretation."""

    @staticmethod
    def shape_circularity(mask: np.ndarray) -> float:
        """Measure how circular the shape is (0=not circular, 1=perfect circle)."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True) + 1e-6
        return float(4.0 * np.pi * (area / (perimeter * perimeter)))

    @staticmethod
    def shape_rectangularity(mask: np.ndarray) -> float:
        """Measure how rectangular the shape is (0=not rectangular, 1=perfect rectangle)."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = w * h
        if bbox_area == 0:
            return 0.0
        return float(area / bbox_area)

    @staticmethod
    def shape_complexity(mask: np.ndarray) -> float:
        """Measure shape complexity (higher = more complex boundary)."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        c = max(cnts, key=cv2.contourArea)

        # Ratio of perimeter to sqrt(area)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if area <= 0:
            return 0.0

        return float(perimeter / (np.sqrt(area) + 1e-6))

    @staticmethod
    def color_variance(hsv: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Measure color variance in HSV space."""
        if mask is not None:
            pixels = hsv[mask > 0]
        else:
            pixels = hsv.reshape(-1, 3)

        if len(pixels) == 0:
            return (0.0, 0.0, 0.0)

        h_var = float(np.var(pixels[:, 0]))
        s_var = float(np.var(pixels[:, 1]))
        v_var = float(np.var(pixels[:, 2]))

        return (h_var, s_var, v_var)

    @staticmethod
    def texture_complexity(gray: np.ndarray) -> float:
        """Measure texture complexity using Laplacian variance."""
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        return float(lap.var())

    @staticmethod
    def aspect_ratio(mask: np.ndarray) -> float:
        """Get width/height ratio of bounding box."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 1.0
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            return 1.0
        return float(w) / float(h)


class AnonymousClassifier:
    """
    Classifies objects into generic visual categories without semantic labels.
    Categories are based purely on visual properties:
    - Shape: circular, rectangular, irregular
    - Texture: smooth, textured, complex
    - Color: uniform, varied
    """

    def __init__(self):
        self.extractor = VisualFeatureExtractor()

    def classify(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Classify crop into generic visual category.

        Returns:
            (category, confidence) where category is like:
            - "circular_smooth"
            - "rectangular_textured"
            - "irregular_uniform"
            etc.
        """
        if crop_bgr.size == 0:
            return ("unknown", 0.0)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = _create_mask(gray)

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # Extract features
        circularity = self.extractor.shape_circularity(mask)
        rectangularity = self.extractor.shape_rectangularity(mask)
        complexity = self.extractor.shape_complexity(mask)
        aspect = self.extractor.aspect_ratio(mask)

        h_var, s_var, v_var = self.extractor.color_variance(hsv, mask)
        color_uniformity = 1.0 / (1.0 + (h_var + s_var + v_var) / 3.0)

        texture_var = self.extractor.texture_complexity(gray)
        texture_smoothness = 1.0 / (1.0 + texture_var / 100.0)

        edge_dens = _edge_density(gray)

        # Determine shape category
        if circularity > 0.7:
            shape_cat = "circular"
            shape_conf = circularity
        elif rectangularity > 0.7:
            shape_cat = "rectangular"
            shape_conf = rectangularity
        elif aspect > 2.5 or aspect < 0.4:
            shape_cat = "elongated"
            shape_conf = min(1.0, abs(np.log(aspect)) / 2.0)
        else:
            shape_cat = "irregular"
            shape_conf = 1.0 - max(circularity, rectangularity)

        # Determine texture category
        if texture_smoothness > 0.6:
            texture_cat = "smooth"
            texture_conf = texture_smoothness
        elif edge_dens > 0.1:
            texture_cat = "structured"
            texture_conf = min(1.0, edge_dens / 0.2)
        else:
            texture_cat = "textured"
            texture_conf = 1.0 - texture_smoothness

        # Determine color category
        if color_uniformity > 0.7:
            color_cat = "uniform"
            color_conf = color_uniformity
        else:
            color_cat = "varied"
            color_conf = 1.0 - color_uniformity

        # Combine into category label
        category = f"{shape_cat}_{texture_cat}_{color_cat}"

        # Overall confidence is average of component confidences
        confidence = float((shape_conf + texture_conf + color_conf) / 3.0)

        return (category, confidence)

    def classify_simple(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Simplified classification with fewer categories.

        Returns categories like:
        - "type_A" (circular, smooth)
        - "type_B" (rectangular, structured)
        - "type_C" (irregular)
        """
        full_category, confidence = self.classify(crop_bgr, mask)

        # Map detailed categories to simple types
        if "circular" in full_category and "smooth" in full_category:
            return ("type_A", confidence)
        elif "rectangular" in full_category:
            return ("type_B", confidence)
        elif "elongated" in full_category:
            return ("type_C", confidence)
        else:
            return ("type_D", confidence)

    def extract_feature_vector(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract numerical feature vector for clustering or similarity comparison.

        Returns:
            numpy array of shape (10,) with normalized features
        """
        if crop_bgr.size == 0:
            return np.zeros(10, dtype=np.float32)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = _create_mask(gray)

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # Extract features
        circularity = self.extractor.shape_circularity(mask)
        rectangularity = self.extractor.shape_rectangularity(mask)
        complexity = self.extractor.shape_complexity(mask)
        aspect = self.extractor.aspect_ratio(mask)

        h_var, s_var, v_var = self.extractor.color_variance(hsv, mask)
        texture_var = self.extractor.texture_complexity(gray)
        edge_dens = _edge_density(gray)

        # Normalize aspect ratio (log scale)
        aspect_norm = float(np.tanh(np.log(aspect + 1e-6)))

        # Normalize complexity
        complexity_norm = float(np.tanh(complexity / 10.0))

        # Normalize texture
        texture_norm = float(np.tanh(texture_var / 200.0))

        features = np.array([
            circularity,
            rectangularity,
            complexity_norm,
            aspect_norm,
            h_var / 100.0,  # normalize color variance
            s_var / 100.0,
            v_var / 100.0,
            texture_norm,
            edge_dens,
            (h_var + s_var + v_var) / 300.0  # total color variation
        ], dtype=np.float32)

        return features


def classify_object(crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None,
                    simple: bool = True) -> Tuple[str, float]:
    """
    Convenience function for anonymous classification.

    Args:
        crop_bgr: BGR image crop of object
        mask: Optional binary mask
        simple: If True, use simple categories (type_A, type_B, etc.)
                If False, use detailed categories

    Returns:
        (category, confidence)
    """
    classifier = AnonymousClassifier()
    if simple:
        return classifier.classify_simple(crop_bgr, mask)
    else:
        return classifier.classify(crop_bgr, mask)