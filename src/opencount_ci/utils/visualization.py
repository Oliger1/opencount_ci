# src/opencount_ci/utils/visualization.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

BBox = Tuple[int, int, int, int]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _clamp_box(box: BBox, W: int, H: int) -> Optional[BBox]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


class VisualizationUtils:
    @staticmethod
    def annotate_advanced(
        image_path: str,
        boxes: List[BBox],
        out_path: Optional[str] = None,
        line_width: int = 2,
        show_count: bool = True,
        font_size: int = 18,
    ) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        W, H = img.size

        colors = [
            (255, 0, 0),
            (0, 200, 0),
            (0, 0, 255),
            (255, 200, 0),
            (255, 0, 255),
            (0, 200, 200),
            (255, 128, 0),
            (128, 0, 255),
        ]

        font = _load_font(font_size)
        font_big = _load_font(font_size + 8)
        pad = 4  # padding për background-in e tekstit

        for i, box in enumerate(boxes):
            clamped = _clamp_box(box, W, H)
            if clamped is None:
                continue
            x1, y1, x2, y2 = clamped
            color = colors[i % len(colors)]

            # vizato kutinë
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            if show_count:
                label = str(i + 1)
                # madhësia e tekstit (prefero textbbox)
                try:
                    tb = draw.textbbox((0, 0), label, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                except Exception:
                    tw, th = draw.textsize(label, font=font)
                if tw <= 0 or th <= 0:
                    continue  # s’ka sens të vizatojmë etiketë

                # provo “mbi kutinë”; nëse s’ka vend, vizato “poshtë”
                y_above = y1 - th - 2 * pad
                if y_above >= 0:
                    text_y0 = y_above
                else:
                    text_y0 = min(H - th - 2 * pad, y1 + 2)

                # clamp brenda imazhit
                text_x0 = max(0, min(x1, W - tw - 2 * pad))
                text_y0 = max(0, min(text_y0, H - th - 2 * pad))
                text_x1 = text_x0 + tw + 2 * pad
                text_y1 = text_y0 + th + 2 * pad

                # sigurim rendi (Pillow kërkon y1 >= y0)
                if text_y1 < text_y0:
                    text_y1 = text_y0 + th + 2 * pad
                if text_x1 < text_x0:
                    text_x1 = text_x0 + tw + 2 * pad

                # background + tekst
                draw.rectangle([text_x0, text_y0, text_x1, text_y1], fill=color)
                draw.text((text_x0 + pad, text_y0 + pad), label, fill=(255, 255, 255), font=font)

        if show_count:
            txt = f"Total: {len(boxes)}"
            try:
                tb2 = draw.textbbox((0, 0), txt, font=font_big)
                tw2, th2 = tb2[2] - tb2[0], tb2[3] - tb2[1]
            except Exception:
                tw2, th2 = draw.textsize(txt, font=font_big)
            bg_w, bg_h = tw2 + 2 * pad, th2 + 2 * pad
            draw.rectangle([0, 0, bg_w, bg_h], fill=(0, 0, 0))
            draw.text((pad, pad), txt, fill=(255, 255, 255), font=font_big)

        if out_path:
            img.save(out_path, quality=95)
        return img

    @staticmethod
    def create_analysis_report(result: Dict[str, Any]) -> str:
        ci = result["confidence_interval"]
        stats = result["statistics"]
        lines = []
        lines.append("OPENCOUNT CI REPORT")
        lines.append("")
        lines.append("Basic")
        lines.append(f"- Image: {Path(result['image']).name}")
        lines.append(f"- Mode: {result['mode']}")
        lines.append(f"- Shape: {result['image_shape'][1]}x{result['image_shape'][0]} px")
        lines.append("")
        lines.append("Results")
        lines.append(f"- Count (median): {result['count']}")
        lines.append(f"- Confidence interval ({int((1-result['alpha'])*100)}%): [{ci[0]:.1f}, {ci[1]:.1f}]")
        lines.append(f"- Base count: {result['base_count']}")
        lines.append("")
        lines.append("Statistics")
        lines.append(f"- Mean: {result['count_mean']:.2f}")
        lines.append(f"- Std:  {stats['std']:.2f}")
        lines.append(f"- CV:   {stats['cv']:.3f}")
        lines.append(f"- Iterations: {result['iterations']}")
        lines.append(f"- Processing time: {result['processing_time']:.2f}s")
        return "\n".join(lines)
