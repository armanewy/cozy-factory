import argparse
import random
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pad_square import pad_square
from tools.art_post.stroke_and_bleed import apply_stroke_and_bleed
from seed_from_id import seed_from_id


PALETTE = {
    "outline": (42, 36, 32, 255),
    "gold": (233, 186, 89, 255),
    "amber": (212, 152, 60, 255),
    "soil": (193, 123, 74, 255),
    "soil_dark": (156, 92, 51, 255),
}


def draw_wheat_tile(size: int = 768, seed: int = 0) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Ground tile (rounded rectangle)
    margin = int(size * 0.08)
    x0, y0, x1, y1 = margin, margin, size - margin, size - margin
    r = int(size * 0.06)
    # Draw soil fill
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=PALETTE["soil"])
    # Soil border
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, outline=PALETTE["soil_dark"], width=int(size * 0.01))

    # Wheat rows (3–4 rows of clusters)
    rows = 4
    col_per_row = 4
    row_gap = (y1 - y0) / (rows + 1)

    stem_w = max(3, size // 120)
    head_w = max(3, size // 90)
    for r_i in range(rows):
        y_base = int(y0 + (r_i + 1) * row_gap)
        for c_i in range(col_per_row):
            # Jittered x position inside the tile
            x_span = (x1 - x0) / (col_per_row + 1)
            x_c = int(x0 + (c_i + 1) * x_span + rnd.uniform(-x_span * 0.15, x_span * 0.15))
            stem_h = int(size * 0.22 + rnd.uniform(-size * 0.02, size * 0.02))
            y_top = max(y0 + 10, y_base - stem_h)

            # Stem
            draw.line([(x_c, y_base), (x_c, y_top)], fill=PALETTE["outline"], width=stem_w)

            # Head (simple stacked ovals)
            grains = 6 + rnd.randint(-1, 1)
            g_spacing = max(6, stem_h // 10)
            gx = x_c
            gy = y_top + g_spacing // 2
            for g in range(grains):
                w = head_w + g
                h = head_w + g
                bbox = [gx - w, gy - h, gx + w, gy + h]
                draw.ellipse(bbox, fill=PALETTE["gold"], outline=PALETTE["outline"], width=max(2, head_w // 2))
                gy += g_spacing

    return img


def build(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seed = seed_from_id("wheat_field_001")
    base = draw_wheat_tile(size=768, seed=seed)

    # Pre-pad to leave room for sticker outline
    tmp_pre = Path("temp/wheat_manual_prepad.png")
    base.save(tmp_pre)
    tmp_pre2 = Path("temp/wheat_manual_prepad2.png")
    pad_square(str(tmp_pre), str(tmp_pre2), padding=36)
    padded = Image.open(tmp_pre2).convert("RGBA")

    # White outline sticker
    sticker = apply_stroke_and_bleed(
        padded,
        bleed_radius=2,
        stroke_px=28,
        stroke_rgb=(255, 255, 255),
        stroke_alpha=255,
        clean_open_px=1,
        clean_close_px=2,
    )

    # Final pad for UI margin
    tmp_final = Path("temp/wheat_manual_final.png")
    sticker.save(tmp_final)
    pad_square(str(tmp_final), str(out_path), padding=64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/art/cards/wheat_field_001.png")
    args = ap.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
