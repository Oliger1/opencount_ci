# src/opencount_ci/api.py
"""Public API with anonymous classification support."""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict

from opencount_ci.analyzer import ObjectCounter
from opencount_ci.config import DetectionConfig, AnalysisConfig

_counter: Optional[ObjectCounter] = None


def _get_counter() -> ObjectCounter:
    """Get or create singleton counter instance."""
    global _counter
    if _counter is None:
        _counter = ObjectCounter()
    return _counter


def configure_detection(**kwargs) -> None:
    """
    Update detection configuration at runtime.

    Parameters
    ----------
    min_area : int
        Minimum object area in pixels
    peak_rel : float
        Watershed peak relative threshold (0-1)
    log_threshold : float
        LoG normalized threshold
    iou_threshold : float
        NMS IoU threshold
    max_image_size : int
        Maximum image dimension

    Examples
    --------
    ::

        configure_detection(min_area=120, peak_rel=0.4, log_threshold=0.05)
    """
    c = _get_counter()
    for k, v in kwargs.items():
        if hasattr(c.detection_config, k):
            setattr(c.detection_config, k, v)
        else:
            raise ValueError(f"Unknown detection parameter: {k}")


def configure_analysis(**kwargs) -> None:
    """
    Update analysis configuration at runtime.

    Parameters
    ----------
    default_iterations : int
        Default bootstrap iterations
    confidence_level : float
        Confidence level (e.g., 0.90 for 90%)
    enable_parallel : bool
        Enable parallel bootstrap
    max_workers : int
        Maximum parallel workers

    Examples
    --------
    ::

        configure_analysis(default_iterations=100, confidence_level=0.95)
    """
    c = _get_counter()
    for k, v in kwargs.items():
        if hasattr(c.analysis_config, k):
            setattr(c.analysis_config, k, v)
        else:
            raise ValueError(f"Unknown analysis parameter: {k}")


def get_configs(as_dict: bool = False) -> Tuple[DetectionConfig | Dict[str, Any],
                                                AnalysisConfig | Dict[str, Any]]:
    """
    Get current configuration.

    Parameters
    ----------
    as_dict : bool
        Return as dictionaries instead of dataclass objects

    Returns
    -------
    tuple
        (DetectionConfig, AnalysisConfig) or (dict, dict)

    Examples
    --------
    ::

        det_cfg, ana_cfg = get_configs(as_dict=True)
        print(det_cfg['min_area'])
    """
    c = _get_counter()
    if as_dict:
        return asdict(c.detection_config), asdict(c.analysis_config)
    return c.detection_config, c.analysis_config


def count_objects(image_path: str, mode: str = "auto") -> int:
    """
    Fast count without bootstrap/statistics.

    Parameters
    ----------
    image_path : str
        Path to image file
    mode : str
        Detection mode: "auto", "watershed", or "log"

    Returns
    -------
    int
        Number of detected objects

    Examples
    --------
    Basic usage::

        n = count_objects("photo.jpg", mode="auto")
        print(f"Found {n} objects")
    """
    return _get_counter().count(image_path, mode=mode)


def analyze_image(
        image_path: str,
        mode: str = "auto",
        iterations: Optional[int] = None,
        confidence_level: Optional[float] = None,
        parallel: Optional[bool] = None,
        verbose: bool = True,
        do_group: bool = False,
        group_k: Optional[int] = None,
        do_classify: bool = False,
) -> Dict[str, Any]:
    """
    Full analysis with confidence intervals, grouping, and anonymous classification.

    Parameters
    ----------
    image_path : str
        Path to image file
    mode : str
        Detection mode: "auto", "watershed", or "log"
    iterations : int, optional
        Bootstrap iterations (default from config)
    confidence_level : float, optional
        Confidence level, e.g., 0.90 (default from config)
    parallel : bool, optional
        Enable parallel bootstrap (default from config)
    verbose : bool
        Print progress messages
    do_group : bool
        Perform anonymous clustering of detected objects
    group_k : int, optional
        Number of clusters (auto-selected if None)
    do_classify : bool
        Apply anonymous visual pattern classification to detected objects

    Returns
    -------
    dict
        Analysis results with keys:
        - count: int, median count
        - confidence_interval: tuple of (lower, upper)
        - base_count: int, count without perturbation
        - count_mean: float, mean bootstrap count
        - statistics: dict with std and cv
        - bootstrap_samples: list of int
        - processing_time: float, seconds
        - boxes: list of (x1,y1,x2,y2) if do_group or do_classify
        - groups: clustering results if do_group
        - labels: anonymous pattern classification if do_classify

    Examples
    --------
    Basic analysis::

        result = analyze_image("photo.jpg", iterations=100, do_group=True, do_classify=True)
        print(f"Count: {result['count']}")
        print(f"Groups: {result['groups']['k']} clusters")
        print(f"Patterns: {[l['label'] for l in result['labels']]}")
    """
    return _get_counter().analyze(
        image_path=image_path,
        mode=mode,
        iterations=iterations,
        confidence_level=confidence_level,
        parallel=parallel,
        verbose=verbose,
        do_group=do_group,
        group_k=group_k,
        do_classify=do_classify,
    )


def batch_analyze(images: List[str], **kwargs) -> List[Dict[str, Any]]:
    """
    Analyze multiple images with same parameters.

    Parameters
    ----------
    images : list of str
        Paths to image files
    **kwargs
        Same as analyze_image

    Returns
    -------
    list of dict
        Analysis results for each image

    Examples
    --------
    ::

        results = batch_analyze(["img1.jpg", "img2.jpg"], iterations=50, do_classify=True)
        for r in results:
            print(f"{r['image']}: {r['count']} objects")
    """
    return _get_counter().batch_analyze(images, **kwargs)


def detect_boxes(image_path: str, mode: str = "auto") -> List[Tuple[int, int, int, int]]:
    """
    Get raw bounding boxes without analysis.

    Parameters
    ----------
    image_path : str
        Path to image file
    mode : str
        Detection mode: "auto", "watershed", or "log"

    Returns
    -------
    list of tuple
        List of (x1, y1, x2, y2) tuples

    Examples
    --------
    ::

        boxes = detect_boxes("image.jpg")
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            print(f"Box {i}: ({x1},{y1}) to ({x2},{y2})")
    """
    counter = _get_counter()
    img = counter.image_processor.load_image(
        image_path,
        counter.detection_config.max_image_size
    )
    img = counter.image_processor.enhance(img)

    if mode == "auto":
        boxes = counter._detect_auto(img)
    else:
        boxes = counter._get_detector(mode).detect(img)

    return boxes


def classify_boxes(image_path: str, boxes: Optional[List[Tuple[int, int, int, int]]] = None) -> List[Dict[str, Any]]:
    """
    Classify objects in bounding boxes using anonymous visual patterns.

    Parameters
    ----------
    image_path : str
        Path to image file
    boxes : list of tuple, optional
        List of (x1, y1, x2, y2). If None, will detect boxes first

    Returns
    -------
    list of dict
        Each dict contains "label" and "confidence" keys.
        Labels are anonymous visual patterns like:
        - pattern_circular_smooth
        - pattern_rectangular_structured
        - pattern_irregular_textured
        - pattern_vertical_elongated
        - pattern_horizontal_elongated
        - pattern_uniform_color
        - pattern_varied_color
        - pattern_small_compact
        - pattern_large_complex
        - pattern_generic

    Examples
    --------
    ::

        labels = classify_boxes("photo.jpg")
        for i, l in enumerate(labels):
            print(f"Object {i}: {l['label']} (conf: {l['confidence']:.2f})")
    """
    counter = _get_counter()
    classifier = counter._get_classifier()

    if classifier is None:
        raise ImportError("Classification module not available")

    # Load image
    import cv2
    img_gray = counter.image_processor.load_image(
        image_path,
        counter.detection_config.max_image_size
    )
    img_gray = counter.image_processor.enhance(img_gray)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Get boxes if not provided
    if boxes is None:
        boxes = counter._detect_auto(img_gray)

    # Classify each box with anonymous patterns
    labels = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = img_bgr[max(0, y1):min(img_bgr.shape[0], y2),
               max(0, x1):min(img_bgr.shape[1], x2)]
        pattern, conf = classifier.classify(crop)
        labels.append({"label": pattern, "confidence": float(conf)})

    return labels


def group_boxes_api(image_path: str, boxes: Optional[List[Tuple[int, int, int, int]]] = None,
                    k: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform anonymous clustering on bounding boxes.

    Parameters
    ----------
    image_path : str
        Path to image file
    boxes : list of tuple, optional
        List of (x1, y1, x2, y2). If None, will detect boxes first
    k : int, optional
        Number of clusters (auto-selected if None)

    Returns
    -------
    dict
        Dictionary with keys:
        - labels: cluster assignment for each box (e.g., [0, 1, 0, 2, ...])
        - k: number of clusters
        - inertias: clustering quality metrics
        - silhouette: cluster separation score

    Examples
    --------
    ::

        result = group_boxes_api("image.jpg", k=3)
        print(f"Grouped into {result['k']} clusters")
        print(f"Cluster assignments: {result['labels']}")
    """
    counter = _get_counter()
    grouping = counter._get_grouping_module()

    if grouping is None:
        raise ImportError("Grouping module not available")

    # Load image
    import cv2
    img_gray = counter.image_processor.load_image(
        image_path,
        counter.detection_config.max_image_size
    )
    img_gray = counter.image_processor.enhance(img_gray)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Get boxes if not provided
    if boxes is None:
        boxes = counter._detect_auto(img_gray)

    # Perform grouping
    return grouping(img_bgr, boxes, k=k)