# src/opencount_ci/classification/__init__.py
"""
Anonymous object classification module.
Uses visual patterns only - NO semantic labels like 'apple', 'car', etc.
"""

from opencount_ci.classification.rules import Rule, ShapeDetector, ColorDetector, TextureDetector
from opencount_ci.classification.concrete_rules import (
    EnhancedRuleClassifier,
    CircularSmoothRule,
    RectangularStructuredRule,
    IrregularTexturedRule,
    VerticalElongatedRule,
    HorizontalElongatedRule,
    UniformColorRule,
    VariedColorRule,
    SmallCompactRule,
    LargeComplexRule,
)
from opencount_ci.classification.classify import (
    AnonymousClassifier,
    VisualFeatureExtractor,
    classify_object,
)

__all__ = [
    # Base classes
    "Rule",
    "ShapeDetector",
    "ColorDetector",
    "TextureDetector",

    # Concrete pattern rules
    "CircularSmoothRule",
    "RectangularStructuredRule",
    "IrregularTexturedRule",
    "VerticalElongatedRule",
    "HorizontalElongatedRule",
    "UniformColorRule",
    "VariedColorRule",
    "SmallCompactRule",
    "LargeComplexRule",

    # Classifiers
    "EnhancedRuleClassifier",
    "AnonymousClassifier",
    "VisualFeatureExtractor",

    # Convenience functions
    "classify_object",
]