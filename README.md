# OpenCount 

**Class-agnostic object counting with confidence intervals, anonymous grouping, and visual pattern classification.**

## Overview

OpenCount CI is a Python library for counting objects in images without requiring labeled training data or semantic object knowledge. It uses:

- **Multi-scale detection**: Watershed and LoG (Laplacian of Gaussian) detectors
- **Bootstrap confidence intervals**: Statistical uncertainty quantification
- **Anonymous clustering**: Group similar objects without semantic labels
- **Visual pattern classification**: Detect patterns based on shape, texture, and color - NO semantic object names

## Key Features

✅ **No training data required** - works out of the box  
✅ **No semantic labels** - purely visual pattern analysis  
✅ **Statistical confidence** - bootstrap-based uncertainty estimates  
✅ **Anonymous grouping** - cluster similar objects (G0, G1, G2...)  
✅ **Visual patterns** - classify by geometry and appearance only  
✅ **Fast & efficient** - optimized detectors with parallel processing  

## Installation

```bash
pip install opencv-python numpy Pillow
pip install -e .
```

Or install directly:

```bash
pip install opencount-ci-enhanced
```

## Quick Start

### Simple Count

```python
from opencount_ci import count_objects

# Fast count - no bootstrap
count = count_objects("image.jpg", mode="auto")
print(f"Found {count} objects")
```

### Full Analysis with Confidence Intervals

```python
from opencount_ci import analyze_image

result = analyze_image(
    "image.jpg",
    mode="auto",
    iterations=100,
    confidence_level=0.90,
    verbose=True
)

print(f"Count: {result['count']}")
print(f"CI (90%): [{result['confidence_interval'][0]}, {result['confidence_interval'][1]}]")
print(f"Processing time: {result['processing_time']:.2f}s")
```

### Anonymous Grouping

```python
result = analyze_image(
    "image.jpg",
    iterations=50,
    do_group=True,
    group_k=3  # or None for auto
)

# Groups are labeled G0, G1, G2, etc.
print(f"Detected {result['groups']['k']} visual groups")
print(f"Group assignments: {result['groups']['labels']}")
```

### Visual Pattern Classification

```python
result = analyze_image(
    "image.jpg",
    iterations=50,
    do_classify=True
)

# Patterns like: pattern_circular_smooth, pattern_rectangular_structured, etc.
for i, label_info in enumerate(result['labels']):
    print(f"Object {i}: {label_info['label']} (conf: {label_info['confidence']:.2f})")
```

## Pattern Types

The classifier detects **anonymous visual patterns** without semantic meaning:

| Pattern | Description |
|---------|-------------|
| `pattern_circular_smooth` | Circular shape with smooth surface |
| `pattern_rectangular_structured` | Rectangular with structured edges |
| `pattern_irregular_textured` | Irregular shape with varied texture |
| `pattern_vertical_elongated` | Vertically elongated object |
| `pattern_horizontal_elongated` | Horizontally elongated object |
| `pattern_uniform_color` | Object with uniform color |
| `pattern_varied_color` | Object with multiple colors |
| `pattern_small_compact` | Small, compact object |
| `pattern_large_complex` | Large object with complex structure |
| `pattern_generic` | No specific pattern detected |

**Note**: These are purely visual descriptors, NOT semantic object names.

## Command Line Interface

```bash
# Simple count
opencount_ci image.jpg

# Full analysis
opencount_ci image.jpg --analysis --iterations 100 --confidence 0.95

# With grouping
opencount_ci image.jpg --analysis --group --k 4

# With visual pattern classification
opencount_ci image.jpg --analysis --classify

# Batch processing
opencount_ci *.jpg --batch --analysis --output results/

# Save annotated image
opencount_ci image.jpg --save output.jpg

# Multiple options
opencount_ci image.jpg --analysis --iterations 100 --group --classify --save annotated.jpg
```

### CLI Options

```
--mode {auto,watershed,log}    Detection mode (default: auto)
--analysis                     Run bootstrap CI analysis
--iterations N                 Bootstrap iterations (default: 50)
--confidence LEVEL             Confidence level, e.g., 0.90
--group                        Anonymous clustering
--k N                          Number of clusters (auto if omitted)
--classify                     Visual pattern classification
--output DIR                   Output directory for reports
--save FILE                    Save annotated image
--format {json,csv,txt}        Report format
--verbose                      Verbose output
```

## Advanced Usage

### Custom Detection Configuration

```python
from opencount_ci import configure_detection, count_objects

# Adjust detection sensitivity
configure_detection(
    min_area=100,           # Minimum object size (pixels)
    peak_rel=0.4,           # Watershed sensitivity
    log_threshold=0.05,     # LoG detection threshold
    iou_threshold=0.5       # NMS overlap threshold
)

count = count_objects("image.jpg")
```

### Batch Analysis

```python
from opencount_ci import batch_analyze

images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = batch_analyze(
    images,
    iterations=100,
    do_classify=True,
    verbose=True
)

for r in results:
    print(f"{r['image']}: {r['count']} objects")
```

### Direct Classification API

```python
from opencount_ci import classify_boxes, detect_boxes

# Detect objects
boxes = detect_boxes("image.jpg", mode="auto")

# Classify with visual patterns
labels = classify_boxes("image.jpg", boxes)

for i, (box, label_info) in enumerate(zip(boxes, labels)):
    print(f"Object {i} at {box}: {label_info['label']} ({label_info['confidence']:.2f})")
```

## Detection Modes

- **`auto`** (default): Combines watershed and LoG with NMS fusion
- **`watershed`**: Marker-controlled watershed segmentation
- **`log`**: Multi-scale Laplacian of Gaussian blob detection

## How It Works

### 1. Detection Pipeline

```
Image → Preprocessing (CLAHE) → Detection (Watershed + LoG) → NMS → Boxes
```

### 2. Bootstrap Confidence Intervals

```
Base Image → Perturb (noise, gamma, blur) → Detect → Count
                ↓
         Repeat N times → Distribution → Percentiles → CI
```

### 3. Anonymous Grouping

```
Boxes → Extract Features (color, shape, texture) → KMeans → Cluster Labels
```

### 4. Visual Pattern Classification

```
Crop → Analyze Shape + Texture + Color → Match Pattern Rules → Label
```

## Configuration

### Detection Parameters

```python
DetectionConfig(
    min_area=64,                    # Min object area (px²)
    peak_rel=0.35,                  # Watershed peak threshold
    log_threshold=0.04,             # LoG detection threshold
    log_sigmas=(2,3,4,6,8),        # LoG scales
    iou_threshold=0.5,              # NMS IoU threshold
    max_image_size=2048,            # Max image dimension
    noise_std_range=(0.01, 0.03),   # Bootstrap noise range
    gamma_range=(0.85, 1.15),       # Bootstrap gamma range
    blur_probability=0.3            # Bootstrap blur chance
)
```

### Analysis Parameters

```python
AnalysisConfig(
    default_iterations=50,          # Bootstrap iterations
    confidence_level=0.90,          # CI level (90%)
    enable_parallel=True,           # Parallel processing
    max_workers=None                # Auto CPU count
)
```

## Output Format

### Analysis Result Dictionary

```python
{
    "image": "path/to/image.jpg",
    "image_shape": (height, width),
    "mode": "auto",
    "count": 42,                           # Median bootstrap count
    "confidence_interval": (38.0, 46.0),   # CI bounds
    "base_count": 41,                      # Count without perturbation
    "count_mean": 42.3,
    "statistics": {
        "std": 2.1,
        "cv": 0.050
    },
    "iterations": 100,
    "alpha": 0.10,
    "bootstrap_samples": [40, 41, 42, ...],
    "processing_time": 3.45,
    "boxes": [(x1,y1,x2,y2), ...],        # If grouping/classification
    "groups": {                             # If do_group=True
        "labels": [0, 1, 0, 2, ...],
        "k": 3,
        "silhouette": 0.65
    },
    "labels": [                             # If do_classify=True
        {"label": "pattern_circular_smooth", "confidence": 0.85},
        {"label": "pattern_rectangular_structured", "confidence": 0.72},
        ...
    ]
}
```

## Examples

### Count Objects with Uncertainty

```python
from opencount_ci import analyze_image

result = analyze_image("coins.jpg", iterations=100)
print(f"Count: {result['count']} ± {result['statistics']['std']:.1f}")
print(f"90% CI: [{result['confidence_interval'][0]}, {result['confidence_interval'][1]}]")
```

### Group Similar Objects

```python
result = analyze_image("mixed_objects.jpg", do_group=True, group_k=4)

from collections import Counter
groups = Counter(result['groups']['labels'])
for gid, count in groups.items():
    print(f"Group {gid}: {count} objects")
```

### Analyze Visual Patterns

```python
result = analyze_image("image.jpg", do_classify=True)

from collections import Counter
patterns = Counter(l['label'] for l in result['labels'])
for pattern, count in patterns.most_common():
    print(f"{pattern}: {count} objects")
```

## Performance Tips

1. **Reduce image size**: Use `max_image_size=1024` for faster processing
2. **Fewer iterations**: Use `iterations=20` for quick estimates
3. **Disable parallel**: Use `parallel=False` on small images
4. **Adjust detection**: Tune `min_area` to filter small noise

## Requirements

- Python >= 3.7
- numpy >= 1.19.0
- opencv-python >= 4.5.0
- Pillow >= 8.0.0

## License

MIT License

## Citation

If you use OpenCount CI in your research:

```bibtex
@software{opencount_ci,
  title={OpenCount CI: Class-Agnostic Object Counting with Confidence Intervals},
  author={OpenCount CI Contributors},
  year={2024},
  url={https://github.com/Oliger1/opencount-ci}
}
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests
4. Submit a pull request

## Support

- Issues: https://github.com/yourusername/opencount-ci/issues
- Docs: https://github.com/yourusername/opencount-ci/blob/main/README.md
