"""Repair clipped sticker borders by adding padding and reconstructing the
outer white edge into the new margin.

This utility is conservative: it keeps the original pixels untouched and only
paints into the newly-added padding area where the original alpha touched the
canvas edge. It works well when a white sticker border was cropped by the
original canvas bounds.

Usage:
    python tools/art_post/fix_clipped_sticker.py <src.png> [dst.png]

Options:
    --pad N       : transparent padding to add on all sides (default 32)
    --grow N      : how far to extend the silhouette into padding (default 24)
    --color R,G,B : fill color for the reconstructed edge (default 255,255,255)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageOps


def _ensure_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")


def _parse_rgb(value: str) -> Tuple[int, int, int]:
    parts = [int(v) for v in value.split(",")]
    if len(parts) != 3 or any(p < 0 or p > 255 for p in parts):
        raise argparse.ArgumentTypeError("--color must be 'R,G,B' (0-255)")
    return tuple(parts)  # type: ignore[return-value]


def repair(src: Path, dst: Path | None, pad: int, grow: int, color: Tuple[int, int, int]) -> Path:
    src = Path(src)
    out = Path(dst) if dst else src
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)

    base = _ensure_rgba(Image.open(src))
    w, h = base.size

    # Add transparent padding around the original content
    padded = ImageOps.expand(base, border=pad, fill=(0, 0, 0, 0))
    pw, ph = padded.size

    # Build masks on the padded image
    alpha = padded.getchannel("A")

    # Detect whether original content touched any edge of the original canvas
    touches = {
        "top": any(alpha.getpixel((x, pad)) > 0 for x in range(pad, pad + w)),
        "bottom": any(alpha.getpixel((x, pad + h - 1)) > 0 for x in range(pad, pad + w)),
        "left": any(alpha.getpixel((pad, y)) > 0 for y in range(pad, pad + h)),
        "right": any(alpha.getpixel((pad + w - 1, y)) > 0 for y in range(pad, pad + h)),
    }

    # If nothing touched an edge, just save with padding and exit
    if not any(touches.values()):
        padded.save(out, format="PNG")
        return out

    # Grow the alpha outward
    grow = max(1, grow)
    k = max(3, grow * 2 + 1)
    expanded = alpha.filter(ImageFilter.MaxFilter(k))
    ring = ImageChops.subtract(expanded, alpha)

    # Constrain the ring to only the outside area beyond the original bounds
    outside = Image.new("L", (pw, ph), 0)
    if touches["top"]:
        for y in range(0, pad):
            outside.paste(255, box=(0, y, pw, y + 1))
    if touches["bottom"]:
        for y in range(pad + h, ph):
            outside.paste(255, box=(0, y, pw, y + 1))
    if touches["left"]:
        for x in range(0, pad):
            outside.paste(255, box=(x, 0, x + 1, ph))
    if touches["right"]:
        for x in range(pad + w, pw):
            outside.paste(255, box=(x, 0, x + 1, ph))

    ring_outside = ImageChops.multiply(ring, outside)

    # Paint the reconstructed edge as solid white (or chosen color)
    layer = Image.new("RGBA", (pw, ph), color + (255,))
    layer.putalpha(ring_outside)

    # Composite: reconstructed edge below the original pixels
    base_bg = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    comp = Image.alpha_composite(base_bg, layer)
    comp = Image.alpha_composite(comp, padded)

    comp.save(out, format="PNG")
    return out


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Repair clipped sticker borders by padding and reconstructing edges")
    p.add_argument("src", help="Input PNG path")
    p.add_argument("dst", nargs="?", help="Output path (defaults to overwrite src)")
    p.add_argument("--pad", type=int, default=32, help="Transparent padding to add on all sides")
    p.add_argument("--grow", type=int, default=24, help="How far to grow the silhouette into padding")
    p.add_argument("--color", type=_parse_rgb, default=(255, 255, 255), help="Reconstruction color 'R,G,B'")
    args = p.parse_args(list(argv) if argv is not None else None)

    repair(Path(args.src), Path(args.dst) if args.dst else None, pad=max(0, args.pad), grow=max(1, args.grow), color=args.color)


if __name__ == "__main__":
    main()

