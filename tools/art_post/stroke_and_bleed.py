"""Post-processing helpers to add matte bleed and a subtle outer stroke to sprites.

Designed to mitigate dark seams around semi-transparent edges when packing
non-premultiplied PNG art into atlases.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageChops, ImageFilter

# Default stroke configuration (cozy dark brown, semi-opaque)
STROKE_RGB: Tuple[int, int, int] = (42, 36, 32)
STROKE_ALPHA: int = 180
STROKE_PX: int = 3
BLEED_RADIUS: int = 2


def _ensure_rgba(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGBA" else image.convert("RGBA")


def alpha_bleed(image: Image.Image, blur_radius: int = BLEED_RADIUS) -> Image.Image:
    """Fill fully transparent pixels with a blurred matte of the nearest colors."""
    rgba = _ensure_rgba(image)
    alpha = rgba.getchannel("A")
    extrema = alpha.getextrema()
    if not extrema or extrema[0] == 255 or blur_radius <= 0:
        return rgba

    # Blur only the pre-multiplied colors; lets edge colors bleed outward.
    bleed_source = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    bleed_source.paste(rgba, mask=alpha)
    blurred = bleed_source.filter(ImageFilter.BoxBlur(blur_radius))

    transparent_mask = ImageChops.invert(alpha)
    if transparent_mask.getbbox() is None:
        return rgba

    result = rgba.copy()
    result.paste(blurred, mask=transparent_mask)
    result.putalpha(alpha)
    return result


def outer_stroke(
    image: Image.Image,
    px: int = STROKE_PX,
    rgb: Tuple[int, int, int] = STROKE_RGB,
    alpha: int = STROKE_ALPHA,
) -> Image.Image:
    """Grow alpha outward and paint it with a semi-opaque stroke color."""
    if px <= 0 or alpha <= 0:
        return _ensure_rgba(image)

    rgba = _ensure_rgba(image)
    base_alpha = rgba.getchannel("A")
    extrema = base_alpha.getextrema()
    if not extrema or extrema[1] == 0:
        return rgba

    kernel = max(3, px * 2 + 1)
    expanded = base_alpha.filter(ImageFilter.MaxFilter(kernel))
    stroke_mask = ImageChops.subtract(expanded, base_alpha)
    if stroke_mask.getbbox() is None:
        return rgba

    stroke_layer = Image.new("RGBA", rgba.size, rgb + (alpha,))
    stroke_layer.putalpha(stroke_mask)
    return Image.alpha_composite(stroke_layer, rgba)


def apply_stroke_and_bleed(
    image: Image.Image,
    bleed_radius: int = BLEED_RADIUS,
    stroke_px: int = STROKE_PX,
    stroke_rgb: Tuple[int, int, int] = STROKE_RGB,
    stroke_alpha: int = STROKE_ALPHA,
) -> Image.Image:
    """Convenience helper that runs matte bleed first, then the outer stroke."""
    with_bleed = alpha_bleed(image, blur_radius=bleed_radius)
    return outer_stroke(with_bleed, px=stroke_px, rgb=stroke_rgb, alpha=stroke_alpha)


def process(
    src: Path,
    dst: Path | None = None,
    bleed_radius: int = BLEED_RADIUS,
    stroke_px: int = STROKE_PX,
    stroke_rgb: Tuple[int, int, int] = STROKE_RGB,
    stroke_alpha: int = STROKE_ALPHA,
) -> Path:
    """Run the stroke + bleed pipeline on ``src`` and write to ``dst`` (in-place if omitted)."""
    src_path = Path(src)
    if dst is None:
        dst_path = src_path
    else:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(src_path).convert("RGBA")
    processed = apply_stroke_and_bleed(
        image,
        bleed_radius=bleed_radius,
        stroke_px=stroke_px,
        stroke_rgb=stroke_rgb,
        stroke_alpha=stroke_alpha,
    )
    processed.save(dst_path, format="PNG")
    return dst_path


def _parse_rgb(value: str) -> Tuple[int, int, int]:
    parts = [int(v) for v in value.split(",")]
    if len(parts) != 3 or any(not (0 <= v <= 255) for v in parts):
        raise argparse.ArgumentTypeError("rgb must be 'r,g,b' with each value between 0-255")
    return tuple(parts)  # type: ignore[return-value]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add matte bleed + cozy stroke to PNG art assets.")
    parser.add_argument("src", help="Input PNG path")
    parser.add_argument("dst", nargs="?", help="Output path (defaults to in-place overwrite)")
    parser.add_argument("--bleed", type=int, default=BLEED_RADIUS, help="Box blur radius for matte bleed")
    parser.add_argument("--stroke", type=int, default=STROKE_PX, help="Stroke thickness in pixels")
    parser.add_argument(
        "--stroke-rgb",
        type=_parse_rgb,
        default=STROKE_RGB,
        help="Stroke color as 'r,g,b' (0-255)",
    )
    parser.add_argument(
        "--stroke-alpha",
        type=int,
        default=STROKE_ALPHA,
        help="Stroke alpha (0-255)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    process(
        src=Path(args.src),
        dst=Path(args.dst) if args.dst else None,
        bleed_radius=max(0, args.bleed),
        stroke_px=max(0, args.stroke),
        stroke_rgb=args.stroke_rgb,
        stroke_alpha=max(0, min(255, args.stroke_alpha)),
    )


if __name__ == "__main__":
    main()
