# src/opencount_ci/core/geometry.py
from __future__ import annotations
from typing import List, Tuple

BBox = Tuple[int, int, int, int]
Blob = Tuple[int, int, float]


def box_area(b: BBox) -> int:
    """Calculate bounding box area."""
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    """Calculate Intersection over Union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    ua = box_area(a) + box_area(b) - inter
    return inter / float(ua + 1e-9)


def non_max_suppression(boxes: List[BBox], iou_threshold: float = 0.5) -> List[BBox]:
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    if not boxes:
        return []

    ordered = sorted(boxes, key=box_area, reverse=True)
    out: List[BBox] = []

    for b in ordered:
        if all(iou(b, c) < iou_threshold for c in out):
            out.append(b)

    return out


def blob_to_box(blob: Blob, margin_factor: float = 0.5) -> BBox:
    """Convert blob (x, y, radius) to bounding box."""
    x, y, r = blob
    margin = max(4, int(r * margin_factor))
    return (
        int(x - margin),
        int(y - margin),
        int(x + margin),
        int(y + margin)
    )