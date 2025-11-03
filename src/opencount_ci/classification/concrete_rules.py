# src/opencount_ci/classification/concrete_rules.py
"""
Anonymous visual pattern detection rules.
NO semantic labels - only visual characteristics.
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2

from .rules import Rule, ShapeDetector, ColorDetector, TextureDetector, _create_mask, _edge_density


class CircularSmoothRule(Rule):
    """Detects circular objects with smooth surfaces."""

    def __init__(self, name="pattern_circular_smooth", priority=10):
        super().__init__(name, priority)

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        if crop_bgr.size == 0:
            return None

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = _create_mask(gray)

        circ = ShapeDetector.circularity(mask)

        if circ < 0.65:
            return None

        # Check smoothness
        texture_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        smoothness = 1.0 / (1.0 + texture_var / 100.0)

        if smoothness > 0.5:
            score = (circ + smoothness) / 2.0
            return (self.name, float(score))

        return None


class RectangularStructuredRule(Rule):
    """Detects rectangular objects with structured edges."""

    def __init__(self, name="pattern_rectangular_structured", priority=8):
        super().__init__(name, priority)

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        if crop_bgr.size == 0:
            return None

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = _create_mask(gray)

        rect = ShapeDetector.rectangularity(mask)

        if rect < 0.6:
            return None

        # Check for structured edges
        edge_dens = _edge_density(gray)

        if edge_dens > 0.04:
            score = (rect + min(1.0, edge_dens / 0.1)) / 2.0
            return (self.name, float(score))

        return None


class IrregularTexturedRule(Rule):
    """Detects irregular shapes with textured surfaces."""

    def __init__(self, name="pattern_irregular_textured", priority=7):
        super().__init__(name, priority)

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        if crop_bgr.size == 0:
            return None

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = _create_mask(gray)

        shape_name, shape_conf = ShapeDetector.classify_shape(mask)

        if shape_name != 'irregular':
            if shape_conf > 0.7:
                return None

        # Check texture
        texture_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        if texture_var > 100:
            score = min(1.0, texture_var / 200.0)
            return (self.name, float(score))

        return None


class VerticalElongatedRule(Rule):
    """Detects vertically elongated objects."""

    def __init__(self, name="pattern_vertical_elongated", priority=6):
        super().__init__(name, priority)

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        if crop_bgr.size == 0:
            return None

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = _create_mask(gray)

        h, w = gray.shape
        aspect = float(w) / float(h + 1e-6)

        if 0.3 <= aspect <= 0.8 and h >= 30:
            score = 1.0 - abs(aspect - 0.5) / 0.5
            return (self.name, float(score))

        return None


class HorizontalElongatedRule(Rule):
    """Detects horizontally elongated objects."""

    def __init__(self, name="pattern_horizontal_elongated", priority=6):
        super().__init__(name, priority)

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        if crop_bgr.size == 0:
            return None

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = _create_mask(gray)

        h, w = gray.shape
        aspect = float(w) / float(h + 1e-6)

        if aspect >= 2.0 and w >= 30:
            score = min(1.0, (aspect - 2.0) / 3.0 + 0.5)
            return (self.name, float(score))

        return None


class UniformColorRule(Rule):
    """Detects objects with uniform color."""

    def __init__(self, name="pattern_uniform_color", priority=5):
        super().__init__(name, priority)

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        if crop_bgr.size == 0:
            return None

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        if mask is None:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            mask = _create_mask(gray)

        # Check color uniformity
        if mask is not None:
            pixels = hsv[mask > 0]
        else:
            pixels = hsv.reshape(-1, 3)

        if len(pixels) == 0:
            return None

        h_var = float(np.var(pixels[:, 0]))
        s_var = float(np.var(pixels[:, 1]))
        v_var = float(np.var(pixels[:, 2]))

        avg_var = (h_var + s_var + v_var) / 3.0
        uniformity = 1.0 / (1.0 + avg_var / 100.0)

        if uniformity > 0.6:
            return (self.name, float(uniformity))

        return None


class VariedColorRule(Rule):
    """Detects objects with varied/multicolor surfaces."""

    def __init__(self, name="pattern_varied_color", priority=5):
        super().__init__(name, priority)

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        if crop_bgr.size == 0:
            return None

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        if mask is None:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            mask = _create_mask(gray)

        # Check color variation
        if mask is not None:
            pixels = hsv[mask > 0]
        else:
            pixels = hsv.reshape(-1, 3)

        if len(pixels) == 0:
            return None

        h_var = float(np.var(pixels[:, 0]))
        s_var = float(np.var(pixels[:, 1]))

        if h_var > 200 or s_var > 200:
            variation = min(1.0, (h_var + s_var) / 800.0)
            return (self.name, float(variation))

        return None


class SmallCompactRule(Rule):
    """Detects small, compact objects."""

    def __init__(self, name="pattern_small_compact", priority=4):
        super().__init__(name, priority)

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        if crop_bgr.size == 0:
            return None

        h, w = crop_bgr.shape[:2]
        area = h * w

        if area <= 32 * 32:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            if mask is None:
                mask = _create_mask(gray)

            circ = ShapeDetector.circularity(mask)
            rect = ShapeDetector.rectangularity(mask)
            compactness = max(circ, rect)

            if compactness > 0.5:
                return (self.name, float(compactness))

        return None


class LargeComplexRule(Rule):
    """Detects large objects with complex structure."""

    def __init__(self, name="pattern_large_complex", priority=4):
        super().__init__(name, priority)

    def __call__(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[str, float]]:
        if crop_bgr.size == 0:
            return None

        h, w = crop_bgr.shape[:2]
        area = h * w

        if area >= 64 * 64:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            if mask is None:
                mask = _create_mask(gray)

            # Check complexity
            edge_dens = _edge_density(gray)
            texture_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

            complexity = (edge_dens * 10 + texture_var / 100.0) / 2.0

            if complexity > 0.5:
                score = min(1.0, complexity)
                return (self.name, float(score))

        return None


# --------------------------
# ENHANCED CLASSIFIER
# --------------------------
class EnhancedRuleClassifier:
    """
    Rule-based classifier using visual patterns only.
    NO semantic object names - only visual characteristics.
    """

    def __init__(self, rules: Optional[list] = None):
        if rules is None:
            # Default anonymous rule set
            rules = [
                CircularSmoothRule(),
                RectangularStructuredRule(),
                IrregularTexturedRule(),
                VerticalElongatedRule(),
                HorizontalElongatedRule(),
                UniformColorRule(),
                VariedColorRule(),
                SmallCompactRule(),
                LargeComplexRule(),
            ]

        # Sort by priority (highest first)
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)

    def classify(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Classify crop using rule cascade.
        Returns (pattern_label, confidence).
        """
        for rule in self.rules:
            result = rule(crop_bgr, mask)
            if result is not None:
                return result

        # Default fallback - generic object
        return ("pattern_generic", 0.0)

    def classify_batch(self, crops: list, masks: Optional[list] = None) -> list:
        """Classify multiple crops."""
        if masks is None:
            masks = [None] * len(crops)

        return [self.classify(crop, mask) for crop, mask in zip(crops, masks)]

    def get_pattern_summary(self, labels: list) -> dict:
        """
        Get summary statistics of detected patterns.

        Args:
            labels: List of classification results (pattern, confidence)

        Returns:
            Dictionary with pattern counts and statistics
        """
        from collections import Counter

        pattern_names = [label[0] for label in labels]
        confidences = [label[1] for label in labels]

        pattern_counts = Counter(pattern_names)

        return {
            "total_objects": len(labels),
            "unique_patterns": len(pattern_counts),
            "pattern_distribution": dict(pattern_counts),
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "min_confidence": float(np.min(confidences)) if confidences else 0.0,
            "max_confidence": float(np.max(confidences)) if confidences else 0.0,
        }