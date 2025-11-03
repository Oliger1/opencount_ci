# src/opencount_ci/cli.py
from __future__ import annotations

import argparse
import sys
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple, Dict, Any

from opencount_ci.__version__ import __version__
from opencount_ci.api import (
    analyze_image,
    count_objects,
    configure_detection,
    detect_boxes,
    classify_boxes,
)
from opencount_ci.utils.visualization import VisualizationUtils
from opencount_ci.utils.reports import ReportUtils


# -----------------------------
# Helpers
# -----------------------------
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in re.split(r"[,\s]+", s.strip()) if x]


def _parse_float_pair(s: str) -> Tuple[float, float]:
    vals = [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("expected two floats: MIN MAX")
    return vals[0], vals[1]


def _find_images(inputs: Iterable[str], recursive: bool) -> List[str]:
    """
    Gjen imazhe nga inputs (file, dir, glob) në mënyrë robuste.
    - Nëse path ekziston dhe është file, pranoje direkt (pa u varur nga extension-lista).
    - Nëse është dir, bëj glob me pattern-in e extension-eve.
    - Përndryshe trajtoje si pattern glob.
    Jep diagnostikë kur nuk gjendet gjë.
    """
    import glob

    imgs: List[str] = []
    not_found: List[str] = []

    for s in inputs:
        p = Path(s)

        # 1) File real? pranoje direkt
        if p.exists() and p.is_file():
            imgs.append(str(p))
            continue

        # 2) Folder → glob
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            for e in list(_IMG_EXTS) + [x.upper() for x in _IMG_EXTS]:
                for q in p.glob(f"{pattern}{e}"):
                    if q.is_file():
                        imgs.append(str(q))
            continue

        # 3) Pattern glob
        matched = False
        for m in glob.glob(s, recursive=recursive):
            mp = Path(m)
            if mp.is_file():
                imgs.append(str(mp))
                matched = True

        if not matched and not p.exists():
            not_found.append(s)

    imgs = sorted(set(imgs))

    if not imgs:
        if not_found:
            print("ERROR: No images found. Checked patterns/paths:", file=sys.stderr)
            for s in not_found:
                print(f"  - {s}", file=sys.stderr)
        else:
            print("ERROR: No images found.", file=sys.stderr)
        return []

    return imgs


def _anonymize_labels(raw_labels: List[str]) -> Dict[str, str]:
    """
    Map çdo label arbitrar (nga klasifikuesi i brendshëm) në C0, C1, ...
    Kthimi: mapping i qëndrueshëm label→Cx sipas rendit alfabetik.
    """
    uniq = sorted(set(raw_labels))
    mapping = {lbl: f"C{i}" for i, lbl in enumerate(uniq)}
    return mapping


def _write_features_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


# -----------------------------
# CLI
# -----------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "OpenCount CI — class-agnostic object counting with confidence intervals, shape-based grouping, and optional feature export"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick count
  opencount_ci image.jpg

  # Full analysis with CI
  opencount_ci image.jpg --analysis --iterations 100

  # Grouping (unsupervised, p.sh. sipas formës/ngjashmërisë)
  opencount_ci image.jpg --analysis --group --k 3

  # Exporto feature-t e çdo objekti për klustrim jashtë CLI
  opencount_ci image.jpg --analysis --features-out features.csv
        """,
    )

    # Inputs / mode
    parser.add_argument("input", nargs="+", help="Image file(s) or glob pattern(s)")
    parser.add_argument("--mode", choices=["auto", "watershed", "log"], default="auto",
                        help="Detection mode (default: auto)")

    # Analysis / CI
    parser.add_argument("--analysis", action="store_true", help="Run CI analysis (bootstrap)")
    parser.add_argument("--iterations", type=int, default=50, help="Bootstrap iterations (default: 50)")
    parser.add_argument("--confidence", type=float, default=0.90, help="Confidence level, e.g., 0.90 (default: 0.90)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel bootstrap")

    # Detection knobs (shape/texture driven; përdori në core)
    parser.add_argument("--min-area", type=int, help="Minimum area (px)")
    parser.add_argument("--max-area", type=int, help="Maximum area (px)")
    parser.add_argument("--min-circularity", type=float, help="Min circularity (0..1)")
    parser.add_argument("--max-circularity", type=float, help="Max circularity (0..1)")
    parser.add_argument("--min-solidity", type=float, help="Min solidity (0..1)")
    parser.add_argument("--max-solidity", type=float, help="Max solidity (0..1)")
    parser.add_argument("--min-extent", type=float, help="Min extent (area/bbox_area)")
    parser.add_argument("--max-extent", type=float, help="Max extent (area/bbox_area)")
    parser.add_argument("--min-eccentricity", type=float, help="Min eccentricity (0..1)")
    parser.add_argument("--max-eccentricity", type=float, help="Max eccentricity (0..1)")
    parser.add_argument("--merge-distance", type=float, help="Merge components whose centroids are within this px")
    parser.add_argument("--split-touching", action="store_true", help="Enable seed-based split of touching components")
    parser.add_argument("--normalize-scale", type=int, help="Resize longest side to this px before detection")
    parser.add_argument("--texture-suppress", choices=["none", "bilateral", "lbp", "gabor"],
                        help="Pre-filter to reduce fine textures before thresholding")
    parser.add_argument("--log-threshold", type=float, help="LoG normalized threshold")
    parser.add_argument("--peak-rel", type=float, help="Relative peak threshold for distance-transform seeds")

    # Grouping / clustering
    parser.add_argument("--group", action="store_true", help="Group detected boxes into clusters (G0, G1, ...)")
    parser.add_argument("--k", type=int, help="Number of clusters (auto-selected if omitted)")

    # Classification (shape/similarity-based)
    parser.add_argument("--classify", action="store_true",
                        help="Classify objects using internal shape/similarity rules (no semantic names).")
    parser.add_argument("--anonymize-labels", action="store_true",
                        help="If classifier returns any labels, remap them to C0, C1, ... for display.")

    # Feature export (për përdorim downstream)
    parser.add_argument("--features-out", type=str,
                        help="Write per-object features to CSV (if available in analysis result).")

    # Output
    parser.add_argument("--output", "-o", help="Output directory for reports (batch)")
    parser.add_argument("--save", help="Save annotated image (single image mode)")
    parser.add_argument("--report", help="Report path (single image, txt format)")
    parser.add_argument("--format", choices=["json", "csv", "txt"], default="json",
                        help="Report format (default: json)")

    parser.add_argument("--batch", action="store_true", help="Treat inputs as multiple images")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recurse into folders")
    parser.add_argument("--version", action="store_true", help="Print version")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args(argv)

    if args.version:
        print(f"opencount_ci version {__version__}")
        return 0

    # ---------------------------
    # Configure detection runtime
    # ---------------------------
    cfg: Dict[str, Any] = {}
    # shape/texture knobs
    for k in (
        "min_area", "max_area",
        "min_circularity", "max_circularity",
        "min_solidity", "max_solidity",
        "min_extent", "max_extent",
        "min_eccentricity", "max_eccentricity",
        "merge_distance", "normalize_scale",
        "log_threshold", "peak_rel",
    ):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    if args.split_touching:
        cfg["split_touching"] = True
    if args.texture_suppress:
        cfg["texture_suppress"] = args.texture_suppress

    if cfg:
        try:
            configure_detection(**cfg)
            if args.verbose:
                print(f"Detection config updated: {cfg}")
        except TypeError as e:
            print(f"WARNING: some config keys not accepted by core: {e}", file=sys.stderr)

    # ---------------------------
    # Find images
    # ---------------------------
    images = _find_images(args.input, args.recursive)
    if not images:
        return 1
    if args.verbose:
        print(f"Found {len(images)} image(s)")

    # ---------------------------
    # ANALYSIS MODE
    # ---------------------------
    if args.analysis:
        results = []

        for im in images:
            if args.verbose or len(images) > 1:
                print(f"\n{'=' * 60}")
                print(f"Processing: {Path(im).name}")
                print("=" * 60)

            try:
                r = analyze_image(
                    im,
                    mode=args.mode,
                    iterations=args.iterations,
                    confidence_level=args.confidence,
                    parallel=not args.no_parallel,
                    verbose=args.verbose,
                    do_group=args.group,
                    group_k=args.k,
                    do_classify=args.classify,
                )
                results.append(r)
            except Exception as e:
                print(f"ERROR processing {im}: {e}", file=sys.stderr)
                continue

        # (Batch) save reports
        if args.output and results:
            outdir = Path(args.output)
            outdir.mkdir(parents=True, exist_ok=True)

            if args.format == "json":
                for r in results:
                    outp = outdir / (Path(r["image"]).stem + "_result.json")
                    ReportUtils.to_json_enhanced(str(outp), r)
                print(f"\nSaved {len(results)} JSON report(s) to {outdir}")

            elif args.format == "csv":
                outp = outdir / "results.csv"
                ReportUtils.to_csv_summary(results, str(outp))
                print(f"\nSaved CSV summary to {outp}")

            else:  # txt
                for r in results:
                    outp = outdir / (Path(r["image"]).stem + "_report.txt")
                    report = VisualizationUtils.create_analysis_report(r)
                    outp.write_text(report, encoding="utf-8")
                print(f"\nSaved {len(results)} text report(s) to {outdir}")

        # Single-image extras
        if len(results) == 1:
            r = results[0]

            # Dump features (nëse r ka 'features': list[dict])
            if args.features_out and isinstance(r.get("features"), list):
                _write_features_csv(Path(args.features_out), r["features"])
                print(f"Features saved to: {args.features_out}")

            if args.report and args.format == "txt":
                Path(args.report).write_text(
                    VisualizationUtils.create_analysis_report(r),
                    encoding="utf-8"
                )
                print(f"\nReport saved to: {args.report}")

            if args.save:
                boxes = r.get("boxes", detect_boxes(r["image"], mode=args.mode))
                VisualizationUtils.annotate_advanced(
                    r["image"], boxes, args.save, show_count=True
                )
                print(f"Annotated image saved to: {args.save}")

        # Console summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for r in results:
            ci = r["confidence_interval"]
            print(f"{Path(r['image']).name}: {r['count']}  "
                  f"CI[{ci[0]:.1f},{ci[1]:.1f}]  base={r['base_count']}")

            # Grouping info
            if args.group and r.get("groups", {}).get("labels"):
                labels = r["groups"]["labels"]
                freq = Counter(labels)
                pretty = ", ".join(f"G{gid}:{cnt}" for gid, cnt in sorted(freq.items()))
                k_used = r["groups"].get("k", 0)
                sil = r["groups"].get("silhouette", 0)
                print(f"  └─ groups(k={k_used}, silhouette={sil:.2f}): {pretty}")

            # Classification info (anonymized nëse kërkohet)
            if args.classify and r.get("labels"):
                raw = [l["label"] for l in r["labels"] if "label" in l]
                if args.anonymize_labels:
                    mapping = _anonymize_labels(raw)
                    mapped = [mapping[x] for x in raw]
                    label_freq = Counter(mapped)
                else:
                    label_freq = Counter(raw)
                pretty = ", ".join(f"{lbl}:{cnt}" for lbl, cnt in sorted(label_freq.items()))
                print(f"  └─ classes: {pretty}")

        return 0

    # ---------------------------
    # SIMPLE COUNT MODE
    # ---------------------------
    print("\n" + "=" * 60)
    print("QUICK COUNT")
    print("=" * 60)

    for p in images:
        try:
            n = count_objects(p, mode=args.mode)

            # Fail-safe: nëse është >50 dhe nuk janë dhënë filtra, provo konservativisht
            if n > 50 and all(getattr(args, k) is None for k in ["min_area", "min_circularity", "min_solidity"]):
                try:
                    configure_detection(min_area=2000, min_circularity=0.5, min_solidity=0.85)
                    n2 = count_objects(p, mode=args.mode)
                    # rikthe config fillestar nëse ishte vendosur
                    base_cfg = {}
                    for k in ("min_area", "min_circularity", "min_solidity"):
                        v = getattr(args, k, None)
                        if v is not None:
                            base_cfg[k] = v
                    if base_cfg:
                        configure_detection(**base_cfg)
                    if args.verbose:
                        print(f"[fail-safe] {Path(p).name}: {n} → {n2}")
                    n = n2
                except Exception:
                    pass

            print(f"{Path(p).name}: {n} objects")

            # Optional shape-based classification (labels mund të jenë numerikë ose të brendshëm; i anonimizojmë nëse kërkohet)
            if args.classify:
                labels = classify_boxes(p)
                raw = [l.get("label", "C?") for l in labels]
                if args.anonymize_labels:
                    mapping = _anonymize_labels(raw)
                    raw = [mapping[x] for x in raw]
                label_freq = Counter(raw)
                pretty = ", ".join(f"{lbl}:{cnt}" for lbl, cnt in sorted(label_freq.items()))
                print(f"  └─ {pretty}")

        except Exception as e:
            print(f"ERROR: {Path(p).name}: {e}", file=sys.stderr)

    # Save annotated image in simple mode
    if args.save and len(images) == 1:
        try:
            boxes = detect_boxes(images[0], mode=args.mode)
            VisualizationUtils.annotate_advanced(images[0], boxes, args.save, show_count=True)
            print(f"\nAnnotated image saved to: {args.save}")
        except Exception as e:
            print(f"ERROR saving annotation: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
