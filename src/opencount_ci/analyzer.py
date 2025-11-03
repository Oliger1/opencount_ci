# src/opencount_ci/analyzer.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from opencount_ci.config import DetectionConfig, AnalysisConfig
from opencount_ci.core.image_utils import ImageProcessor
from opencount_ci.detectors.watershed import WatershedDetector
from opencount_ci.detectors.log import LoGDetector
from opencount_ci.core.geometry import non_max_suppression

# Optional grouping backend (anonymous clustering of boxes)
_HAS_GROUPING = False
try:
    from .grouping import group_boxes
    _HAS_GROUPING = True
except Exception:
    _HAS_GROUPING = False
    def group_boxes(img_bgr, boxes, k=None, kmin=2, kmax=6, seed=42):
        return {"labels": [], "k": 0, "inertias": {}}

# Optional classification backend (anonymous visual patterns only)
_HAS_CLASSIFICATION = False
try:
    from .classification.concrete_rules import EnhancedRuleClassifier
    _HAS_CLASSIFICATION = True
except Exception:
    _HAS_CLASSIFICATION = False
    EnhancedRuleClassifier = None


class ObjectCounter:
    """
    Main analyzer with:
    - Object detection (watershed/log/auto)
    - Bootstrap confidence intervals
    - Anonymous grouping (clustering)
    - Anonymous classification (visual patterns only)
    """

    def __init__(self,
                 detection_config: Optional[DetectionConfig] = None,
                 analysis_config: Optional[AnalysisConfig] = None):
        self.detection_config = detection_config or DetectionConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
        self.image_processor = ImageProcessor()
        self._classifier = None

    def _get_detector(self, mode: str):
        dc = self.detection_config
        if mode == "watershed":
            return WatershedDetector(
                min_area=dc.min_area,
                peak_rel=dc.peak_rel,
                adaptive_kernel=dc.adaptive_kernel,
                morph_kernel_size=dc.morph_kernel_size,
                morph_iterations=dc.morph_iterations,
                dilation_iterations=dc.dilation_iterations,
            )
        elif mode == "log":
            return LoGDetector(
                log_sigmas=dc.log_sigmas,
                log_threshold=dc.log_threshold,
                blob_margin_factor=dc.blob_margin_factor,
                iou_threshold=dc.iou_threshold,
                min_area=dc.min_area,
                max_blobs=2000,
            )
        elif mode == "auto":
            return None
        else:
            raise ValueError("mode must be one of: auto | watershed | log")

    def _detect_auto(self, image: np.ndarray):
        w = self._get_detector("watershed")
        l = self._get_detector("log")
        boxes_w = w.detect(image)
        boxes_l = l.detect(image)
        boxes = non_max_suppression(
            list(boxes_w) + list(boxes_l),
            self.detection_config.iou_threshold
        )
        return boxes

    def _get_classifier(self):
        """Get or create anonymous classifier instance."""
        if not _HAS_CLASSIFICATION:
            return None
        if self._classifier is None:
            self._classifier = EnhancedRuleClassifier()
        return self._classifier

    def _get_grouping_module(self):
        """Check if grouping module is available."""
        return group_boxes if _HAS_GROUPING else None

    def count(self, image_path: str, mode: str = "auto") -> int:
        """Simple count without analysis."""
        img_gray = self.image_processor.load_image(image_path, self.detection_config.max_image_size)
        img_gray = self.image_processor.enhance(img_gray)

        if mode == "auto":
            boxes = self._detect_auto(img_gray)
        else:
            det = self._get_detector(mode)
            boxes = det.detect(img_gray)

        return int(len(boxes))

    def analyze(self, image_path: str, mode: str = "auto",
                iterations: Optional[int] = None, confidence_level: Optional[float] = None,
                parallel: Optional[bool] = None, verbose: bool = True,
                seed: int = 1337,
                do_group: bool = False, group_k: Optional[int] = None,
                do_classify: bool = False) -> Dict[str, Any]:
        """
        Full analysis with CI, grouping, and anonymous classification.

        Parameters
        ----------
        image_path : str
        mode : {"auto", "watershed", "log"}
        iterations : int, bootstrap iterations
        confidence_level : float in (0,1], e.g. 0.90
        parallel : bool, run bootstrap in parallel
        verbose : bool, print progress
        seed : int, RNG seed
        do_group : bool, perform anonymous clustering
        group_k : Optional[int], number of clusters
        do_classify : bool, perform anonymous visual classification
        """
        t0 = time.time()

        # Configuration
        iterations = int(iterations if iterations is not None else self.analysis_config.default_iterations)
        iterations = max(self.analysis_config.min_iterations, min(self.analysis_config.max_iterations, iterations))
        conf = confidence_level if confidence_level is not None else self.analysis_config.confidence_level
        alpha = 1.0 - float(conf)
        parallel = bool(self.analysis_config.enable_parallel if parallel is None else parallel)

        # Load image
        if verbose:
            print("Loading image...", flush=True)
        img_gray = self.image_processor.load_image(image_path, self.detection_config.max_image_size)
        img_gray = self.image_processor.enhance(img_gray)

        # Base detection
        if verbose:
            print(f"Running base detection (mode={mode})...", flush=True)
        base_boxes = self._detect_auto(img_gray) if mode == "auto" else self._get_detector(mode).detect(img_gray)
        base_count = int(len(base_boxes))
        if verbose:
            print(f"Base detection done. Base count = {base_count}", flush=True)

        # Optional grouping
        base_groups: Dict[str, Any] = {"labels": [], "k": 0, "inertias": {}}
        if do_group and base_boxes:
            import cv2
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            base_groups = group_boxes(img_bgr, base_boxes, k=group_k)

        # Optional anonymous classification
        base_labels: List[Dict[str, Any]] = []
        if do_classify and base_boxes:
            classifier = self._get_classifier()
            if classifier is None:
                if verbose:
                    print("Warning: Classification module not available", flush=True)
            else:
                import cv2
                img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                for box in base_boxes:
                    x1, y1, x2, y2 = box
                    crop = img_bgr[max(0, y1):min(img_bgr.shape[0], y2),
                           max(0, x1):min(img_bgr.shape[1], x2)]
                    pattern, conf = classifier.classify(crop)
                    base_labels.append({"label": pattern, "confidence": float(conf)})

        # Bootstrap worker
        def run_one(i: int) -> int:
            g = self.image_processor.perturb(
                img_gray, seed=seed + i,
                noise_std_range=self.detection_config.noise_std_range,
                gamma_range=self.detection_config.gamma_range,
                brightness_range=self.detection_config.brightness_range,
                blur_probability=self.detection_config.blur_probability,
            )
            if mode == "auto":
                b = self._detect_auto(g)
            else:
                det = self._get_detector(mode)
                b = det.detect(g)
            return int(len(b))

        # Bootstrap
        if iterations <= 1:
            counts = [base_count]
            if verbose:
                print("Bootstrap skipped (iterations <= 1).", flush=True)
        elif parallel:
            max_workers = os.cpu_count() or 4
            if self.analysis_config.max_workers:
                max_workers = min(max_workers, int(self.analysis_config.max_workers))
            if verbose:
                print(f"Starting bootstrap in parallel: {iterations} iters, {max_workers} workers...", flush=True)
            counts = [0] * iterations
            done = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(run_one, i): i for i in range(iterations)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        counts[i] = fut.result()
                    except Exception:
                        counts[i] = 0
                    done += 1
                    if verbose:
                        print(f"\rProgress: {done}/{iterations}", end="", flush=True)
            if verbose:
                print()
        else:
            if verbose:
                print(f"Starting bootstrap sequentially: {iterations} iters...", flush=True)
            counts = []
            for i in range(iterations):
                counts.append(run_one(i))
                if verbose:
                    print(f"\rProcessing {i+1}/{iterations} ...", end="", flush=True)
            if verbose:
                print(f"\rProcessing {iterations}/{iterations} done.    ")

        # Statistics
        arr = np.array(counts, dtype=np.int32)
        median = int(np.median(arr))
        lo = int(np.quantile(arr, alpha / 2.0))
        hi = int(np.quantile(arr, 1.0 - alpha / 2.0))
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        cv = float(std / mean) if mean > 0 else 0.0

        out: Dict[str, Any] = {
            "image": image_path,
            "image_shape": (int(img_gray.shape[0]), int(img_gray.shape[1])),
            "mode": mode,
            "count": median,
            "confidence_interval": (float(lo), float(hi)),
            "base_count": base_count,
            "count_mean": mean,
            "statistics": {"std": std, "cv": cv},
            "iterations": int(len(arr)),
            "alpha": float(alpha),
            "bootstrap_samples": arr.tolist(),
            "processing_time": float(time.time() - t0),
            "detection_config": {
                "min_area": self.detection_config.min_area,
                "peak_rel": self.detection_config.peak_rel,
                "log_threshold": self.detection_config.log_threshold,
            },
            "boxes": base_boxes if (do_group or do_classify) else [],
            "groups": base_groups if do_group else {"labels": [], "k": 0, "inertias": {}},
            "labels": base_labels if do_classify else [],
        }

        if verbose:
            ci_pct = int((1.0 - alpha) * 100)
            print(f"Done in {out['processing_time']:.2f}s  |  "
                  f"count={out['count']}  CI{ci_pct}%=[{lo},{hi}]  base={base_count}", flush=True)

        return out

    def batch_analyze(self, images: List[str], **kwargs) -> List[Dict[str, Any]]:
        return [self.analyze(p, **kwargs) for p in images]