from __future__ import annotations
from typing import Dict, Any, List
import json
from pathlib import Path
import time
import sys
import csv


class ReportUtils:
    @staticmethod
    def to_json_enhanced(path: str, payload: Dict[str, Any]) -> None:
        out = {
            **payload,
            "metadata": {
                "library": "opencount_ci",
                "version": "0.2.0",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    @staticmethod
    def to_csv_summary(results: List[Dict[str, Any]], output_path: str) -> None:
        if not results:
            Path(output_path).write_text("", encoding="utf-8")
            return
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "image",
                    "mode",
                    "count",
                    "ci_lower",
                    "ci_upper",
                    "base_count",
                    "mean",
                    "std",
                    "cv",
                    "processing_time",
                ]
            )
            for r in results:
                ci = r["confidence_interval"]
                w.writerow(
                    [
                        Path(r["image"]).name,
                        r["mode"],
                        r["count"],
                        f"{ci[0]:.1f}",
                        f"{ci[1]:.1f}",
                        r["base_count"],
                        f"{r['count_mean']:.2f}",
                        f"{r['statistics']['std']:.2f}",
                        f"{r['statistics']['cv']:.3f}",
                        f"{r['processing_time']:.2f}",
                    ]
                )
