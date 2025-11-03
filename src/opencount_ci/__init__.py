# src/opencount_ci/__init__.py
from .api import (
    count_objects,
    analyze_image,
    batch_analyze,
    configure_detection,
    configure_analysis,
    get_configs,
    detect_boxes,
    classify_boxes,
    group_boxes_api,
)
from .config import DetectionConfig, AnalysisConfig
from .__version__ import __version__

__all__ = [
    "count_objects",
    "analyze_image",
    "batch_analyze",
    "configure_detection",
    "configure_analysis",
    "get_configs",
    "detect_boxes",
    "classify_boxes",
    "group_boxes_api",
    "DetectionConfig",
    "AnalysisConfig",
    "__version__",
]