"""Enforce a clean rectangular white border around an image.

Strips any irregular sticker/halo by cropping to the bounding box of
"non‑white" pixels (difference-from-white above a tolerance), then places the
cropped art onto a solid white canvas with a fixed rectangular frame.

No alpha manipulation or thresholds on the subject itself, so no black edge
artifacts in fine details.

Result is saved fully opaque (RGB).

Usage:
  python tools/art_post/enforce_rect_white.py src.png [dst.png] --frame 48 --tol 18
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse

from PIL import Image, ImageChops


def enforce(
    src: Path,
    dst: Path | None,
    frame: int = 48,
    tol: int = 18,
    inset: int = 0,
) -> Path:
    src = Path(src)
    out = Path(dst) if dst else src
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src).convert("RGBA")
    # Flatten onto white so we can measure distance from white
    rgb = Image.new("RGB", img.size, (255, 255, 255))
    rgb.paste(img.convert("RGB"), mask=img.split()[-1])

    # Compute a mask of pixels that are sufficiently "non-white"
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, (255, 255, 255)))
    # diff is higher where pixels are farther from white; threshold by tolerance
    tol = max(0, min(255, int(tol)))
    mask = diff.convert("L").point(lambda v: 255 if v > tol else 0)
    bbox = mask.getbbox()
    if bbox is None:
        cropped_rgba = img
    else:
        cropped_rgba = img.crop(bbox)

    # Optional additional inward crop to shave any near-edge imperfections
    if inset > 0:
        inset = int(inset)
        cw0, ch0 = cropped_rgba.size
        left = min(cw0 // 2 - 1, inset)
        top = min(ch0 // 2 - 1, inset)
        right = cw0 - left
        bottom = ch0 - top
        cropped_rgba = cropped_rgba.crop((left, top, right, bottom))

    cw, ch = cropped_rgba.size
    frame = max(0, int(frame))
    canvas = Image.new("RGB", (cw + 2 * frame, ch + 2 * frame), (255, 255, 255))
    canvas.paste(cropped_rgba.convert("RGB"), (frame, frame), mask=cropped_rgba.split()[-1])
    return canvas, (frame, frame, frame + cw, frame + ch)
    
def _snap_border_whites(img: Image.Image, inner_box: tuple[int,int,int,int], snap: int, tol: int) -> Image.Image:
    if snap <= 0:
        return img
    tol = max(0, min(255, tol))
    w, h = img.size
    white = (255, 255, 255)
    x0, y0, x1, y1 = inner_box
    snap = max(1, snap)
    px = img.load()
    def near_white(rgb):
        r,g,b = rgb
        return (abs(255-r)+abs(255-g)+abs(255-b))//3 <= tol
    # Top strip
    for y in range(max(0, y0 - snap), y0):
        for x in range(x0, x1):
            if near_white(px[x, y]):
                px[x, y] = white
    # Bottom strip
    for y in range(y1, min(h, y1 + snap)):
        for x in range(x0, x1):
            if near_white(px[x, y]):
                px[x, y] = white
    # Left strip
    for x in range(max(0, x0 - snap), x0):
        for y in range(y0, y1):
            if near_white(px[x, y]):
                px[x, y] = white
    # Right strip
    for x in range(x1, min(w, x1 + snap)):
        for y in range(y0, y1):
            if near_white(px[x, y]):
                px[x, y] = white
    return img


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Add a clean rectangular white border around an image")
    p.add_argument("src")
    p.add_argument("dst", nargs="?")
    p.add_argument("--frame", type=int, default=48, help="Border thickness on each side in pixels")
    p.add_argument("--tol", type=int, default=18, help="Tolerance (0-255) for distance from white when finding the inner crop bbox")
    p.add_argument("--inset", type=int, default=0, help="Additional inward crop in pixels to shave any near-edge imperfections")
    p.add_argument("--snap", type=int, default=0, help="Width (px) inside the inner edge to snap near-white pixels to pure white")
    p.add_argument("--snap-tol", type=int, default=40, help="Near-white tolerance for snapping (0-255)")
    args = p.parse_args(list(argv) if argv is not None else None)

    out_path = Path(args.dst) if args.dst else None
    canvas, inner = enforce(Path(args.src), out_path, frame=args.frame, tol=args.tol, inset=args.inset)
    canvas = _snap_border_whites(canvas, inner, args.snap, args.snap_tol)
    target = Path(args.dst) if args.dst else Path(args.src)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG")


if __name__ == "__main__":
    main()
