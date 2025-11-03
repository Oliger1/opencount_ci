# src/opencount_ci/config.py
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DetectionConfig:
    """Detection parameters for Watershed and LoG."""

    # Basic area constraints
    min_area: int = 64
    max_area: Optional[int] = None

    # Watershed parameters
    peak_rel: float = 0.35
    morph_kernel_size: int = 3
    morph_iterations: int = 1
    dilation_iterations: int = 2

    # LoG parameters
    log_threshold: float = 0.04
    log_sigmas: Tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 8.0)

    # Fusion / NMS
    iou_threshold: float = 0.5
    blob_margin_factor: float = 0.5

    # Image preprocessing
    max_image_size: int = 2048
    adaptive_kernel: bool = True

    # Shape filters (advanced)
    min_circularity: Optional[float] = None
    max_circularity: Optional[float] = None
    min_solidity: Optional[float] = None
    max_solidity: Optional[float] = None
    min_extent: Optional[float] = None
    max_extent: Optional[float] = None
    min_eccentricity: Optional[float] = None
    max_eccentricity: Optional[float] = None

    # Advanced options
    merge_distance: Optional[float] = None
    split_touching: bool = False
    normalize_scale: Optional[int] = None
    texture_suppress: str = "none"

    # Perturbations for bootstrap
    noise_std_range: Tuple[float, float] = (0.01, 0.03)
    gamma_range: Tuple[float, float] = (0.85, 1.15)
    brightness_range: Tuple[float, float] = (0.95, 1.05)
    blur_probability: float = 0.3

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.min_area <= 0:
            raise ValueError("min_area must be > 0")
        if not (0.0 < self.peak_rel < 1.0):
            raise ValueError("peak_rel must be in (0, 1)")
        if self.log_threshold <= 0:
            raise ValueError("log_threshold must be > 0")
        if not self.log_sigmas:
            raise ValueError("log_sigmas cannot be empty")
        if self.max_area is not None and self.max_area < self.min_area:
            raise ValueError("max_area must be >= min_area")


@dataclass
class AnalysisConfig:
    """Analysis and bootstrap parameters."""

    default_iterations: int = 50
    min_iterations: int = 1
    max_iterations: int = 500
    confidence_level: float = 0.90

    # Performance
    enable_parallel: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 8

    # Output detail
    save_bootstrap_samples: bool = True
    detailed_stats: bool = True

    def alpha(self) -> float:
        """Return alpha for confidence interval."""
        return 1.0 - self.confidence_level

    def validate(self) -> None:
        """Validate analysis configuration."""
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be in (0, 1)")
        if self.default_iterations < self.min_iterations:
            raise ValueError("default_iterations must be >= min_iterations")
        if self.default_iterations > self.max_iterations:
            raise ValueError("default_iterations must be <= max_iterations")