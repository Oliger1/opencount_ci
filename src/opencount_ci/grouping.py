# src/opencount_ci/grouping.py
"""Enhanced grouping with better color and shape discrimination."""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

BBox = Tuple[int, int, int, int]


def _safe_crop(img: np.ndarray, box: BBox) -> np.ndarray:
    """Safely crop image within bounds."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return img[0:1, 0:1].copy()
    return img[y1:y2, x1:x2]


def _extract_color_histogram(crop_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    """Extract normalized color histogram in HSV space."""
    if crop_bgr.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # Histogram for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # Normalize
    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-9)
    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-9)
    hist_v = hist_v.flatten() / (hist_v.sum() + 1e-9)

    return np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)


def _dominant_color(crop_bgr: np.ndarray) -> Tuple[float, float, float]:
    """Extract dominant color in HSV."""
    if crop_bgr.size == 0:
        return (0.0, 0.0, 0.0)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Use median as robust central tendency
    h_dom = float(np.median(pixels[:, 0]))
    s_dom = float(np.median(pixels[:, 1]))
    v_dom = float(np.median(pixels[:, 2]))

    return (h_dom, s_dom, v_dom)


def _shape_features(mask: np.ndarray) -> np.ndarray:
    """Extract comprehensive shape features."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return np.zeros(7, dtype=np.float32)

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True) + 1e-6

    # Circularity
    circularity = 4.0 * np.pi * (area / (perimeter * perimeter))

    # Solidity (convexity)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull) + 1e-6
    solidity = area / hull_area

    # Extent (bounding box fill)
    x, y, w, h = cv2.boundingRect(c)
    extent = area / (w * h + 1e-6)

    # Aspect ratio
    aspect = float(w) / float(h + 1e-6)

    # Moments for orientation and eccentricity
    moments = cv2.moments(c)
    if moments['m00'] > 0:
        mu20 = moments['mu20'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']

        # Eccentricity approximation
        diff = mu20 - mu02
        trace = mu20 + mu02
        eccentricity = np.sqrt(1 - (4 * (mu20 * mu02 - mu11 ** 2)) / (trace ** 2 + 1e-9))
    else:
        eccentricity = 0.0

    # Compactness
    compactness = (perimeter * perimeter) / (area + 1e-6)

    return np.array([
        circularity, solidity, extent, aspect,
        eccentricity, compactness, np.log(area + 1)
    ], dtype=np.float32)


def _texture_features(gray: np.ndarray) -> np.ndarray:
    """Extract texture features using edge and gradient analysis."""
    if gray.size == 0:
        return np.zeros(4, dtype=np.float32)

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (gray.size + 1e-9)

    # Gradient magnitude statistics
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    grad_mean = float(grad_mag.mean())
    grad_std = float(grad_mag.std())

    # Local Binary Pattern approximation (variance-based)
    lbp_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    return np.array([edge_density, grad_mean, grad_std, lbp_var], dtype=np.float32)


def _binary_mask_adaptive(gray: np.ndarray) -> np.ndarray:
    """Create binary mask using adaptive thresholding."""
    if gray.size == 0:
        return np.zeros_like(gray)

    # Try adaptive threshold first
    try:
        block_size = max(3, min(gray.shape) // 8)
        if block_size % 2 == 0:
            block_size += 1
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 2
        )
    except:
        # Fallback to Otsu
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask


def features_for_box(img_bgr: np.ndarray, box: BBox, full_shape: Tuple[int, int]) -> np.ndarray:
    """
    Extract comprehensive feature vector for a bounding box.

    Features (total ~40 dimensions):
    - Color histogram (24): 8 bins × 3 channels (HSV)
    - Dominant color (3): H, S, V medians
    - Color statistics (6): mean/std for H, S, V
    - Shape features (7): circularity, solidity, extent, aspect, eccentricity, compactness, log(area)
    - Texture features (4): edge density, gradient mean/std, LBP variance
    - Spatial features (2): relative position, relative size
    """
    crop = _safe_crop(img_bgr, box)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    h, w = crop.shape[:2]
    H, W = full_shape

    # Spatial features
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0 / max(1, W)
    center_y = (y1 + y2) / 2.0 / max(1, H)
    rel_area = float(w * h) / float(max(1, H * W))

    # Color features
    color_hist = _extract_color_histogram(crop, bins=8)
    dom_h, dom_s, dom_v = _dominant_color(crop)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = hsv.mean(axis=(0, 1))
    h_std, s_std, v_std = hsv.std(axis=(0, 1))

    # Shape features
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = _binary_mask_adaptive(gray)
    shape_feats = _shape_features(mask)

    # Texture features
    texture_feats = _texture_features(gray)

    # Combine all features
    features = np.concatenate([
        color_hist,  # 24
        [dom_h, dom_s, dom_v],  # 3
        [h_mean, s_mean, v_mean, h_std, s_std, v_std],  # 6
        shape_feats,  # 7
        texture_feats,  # 4
        [center_x, center_y, rel_area]  # 3
    ]).astype(np.float32)

    return features


def build_feature_matrix(img_bgr: np.ndarray, boxes: List[BBox]) -> np.ndarray:
    """Build feature matrix for all boxes with robust normalization."""
    H, W = img_bgr.shape[:2]
    feats = [features_for_box(img_bgr, b, (H, W)) for b in boxes]

    if not feats:
        return np.zeros((0, 47), dtype=np.float32)

    X = np.vstack(feats)

    # Robust standardization (median/IQR)
    if len(X) > 1:
        med = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = (q3 - q1)
        iqr[iqr < 1e-6] = 1.0
        X = (X - med) / iqr

    return X


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """KMeans++ initialization for better cluster centers."""
    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)

    # First center: random
    idx = rng.integers(0, n)
    centers[0] = X[idx]

    # Subsequent centers: D^2 weighting
    d2 = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        d2 = np.minimum(d2, ((X - centers[i - 1]) ** 2).sum(axis=1))
        probs = d2 / (d2.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]

    return centers


def _kmeans_fit_predict(X: np.ndarray, k: int, iters: int = 100, seed: int = 42) -> Tuple[
    np.ndarray, float, np.ndarray]:
    """
    KMeans clustering with multiple restarts for stability.
    Returns (labels, inertia, centers).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if k <= 0 or n == 0:
        return np.zeros((n,), dtype=np.int32), 0.0, np.zeros((k, X.shape[1]), dtype=X.dtype)

    if k > n:
        k = n

    # Multiple restarts for stability
    best_labels = None
    best_inertia = np.inf
    best_centers = None

    n_restarts = min(5, max(1, 10 // k))

    for restart in range(n_restarts):
        centers = _kmeans_pp_init(X, k, rng)
        labels = np.zeros((n,), dtype=np.int32)

        for _ in range(iters):
            # Assign to nearest center
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            if np.array_equal(new_labels, labels):
                break

            labels = new_labels

            # Update centers
            for j in range(k):
                mask = (labels == j)
                if np.any(mask):
                    centers[j] = X[mask].mean(axis=0)
                else:
                    # Reinitialize empty cluster
                    centers[j] = X[rng.integers(0, n)]

        # Calculate inertia
        inertia = float(((X - centers[labels]) ** 2).sum())

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_inertia, best_centers


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Calculate simplified silhouette score for cluster quality."""
    n = len(labels)
    if n <= 1:
        return 0.0

    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0.0

    # Pairwise distances
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    scores = []
    for i in range(n):
        label = labels[i]

        # a: mean distance to same cluster
        same_cluster = labels == label
        if same_cluster.sum() > 1:
            a = dists[i, same_cluster].sum() / (same_cluster.sum() - 1)
        else:
            a = 0.0

        # b: min mean distance to other clusters
        b = np.inf
        for other_label in unique_labels:
            if other_label != label:
                other_cluster = labels == other_label
                if other_cluster.any():
                    b_temp = dists[i, other_cluster].mean()
                    b = min(b, b_temp)

        if b == np.inf:
            b = 0.0

        if max(a, b) > 0:
            scores.append((b - a) / max(a, b))
        else:
            scores.append(0.0)

    return float(np.mean(scores))


def _choose_k_auto(X: np.ndarray, kmin: int = 2, kmax: int = 8, seed: int = 42) -> Tuple[int, Dict[int, float]]:
    """
    Automatically choose k using elbow method with silhouette validation.
    """
    if X.shape[0] < kmin:
        return max(1, X.shape[0]), {}

    kmax = min(kmax, X.shape[0])
    inertias: Dict[int, float] = {}
    silhouettes: Dict[int, float] = {}

    for k in range(kmin, kmax + 1):
        labels, inertia, _ = _kmeans_fit_predict(X, k, seed=seed)
        inertias[k] = inertia
        silhouettes[k] = _silhouette_score(X, labels)

    # Find elbow using relative improvement
    best_k = kmin
    prev_inertia = None

    for k in range(kmin, kmax + 1):
        if prev_inertia is not None:
            rel_impr = (prev_inertia - inertias[k]) / (prev_inertia + 1e-9)
            # Stop if improvement < 10% and silhouette is reasonable
            if rel_impr < 0.10 and silhouettes.get(k - 1, 0) > 0.2:
                best_k = k - 1
                break
        prev_inertia = inertias[k]
        best_k = k

    return best_k, inertias


def group_boxes(
        img_bgr: np.ndarray,
        boxes: List[BBox],
        k: Optional[int] = None,
        kmin: int = 2,
        kmax: int = 8,
        seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform enhanced clustering on detected boxes using color, shape, and texture features.

    Parameters
    ----------
    img_bgr : BGR image
    boxes : List of bounding boxes
    k : Number of clusters (auto-selected if None)
    kmin, kmax : Range for automatic k selection
    seed : Random seed

    Returns
    -------
    dict with keys:
        - "labels": cluster assignment for each box
        - "k": number of clusters used
        - "inertias": inertia values for different k
        - "silhouette": cluster quality score
    """
    X = build_feature_matrix(img_bgr, boxes)
    n = X.shape[0]

    if n == 0:
        return {"labels": [], "k": 0, "inertias": {}, "silhouette": 0.0}

    if k is None:
        k, inertias = _choose_k_auto(X, kmin=kmin, kmax=kmax, seed=seed)
    else:
        inertias = {}
        k = max(1, min(int(k), n))

    labels, inertia, _ = _kmeans_fit_predict(X, k, seed=seed)
    inertias[k] = inertia

    silhouette = _silhouette_score(X, labels)

    return {
        "labels": labels.tolist(),
        "k": int(k),
        "inertias": inertias,
        "silhouette": float(silhouette)
    }