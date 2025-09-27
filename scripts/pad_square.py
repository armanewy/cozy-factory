# scripts/pad_square.py
from PIL import Image
import argparse
import os

def pad_square(input_path: str, output_path: str, padding: int = 64, background=(0,0,0,0)):
    """
    Pads an image to a square canvas centered with transparent background by default.
    """
    img = Image.open(input_path).convert("RGBA")
    w, h = img.size
    side = max(w, h) + (padding * 2)

    canvas = Image.new("RGBA", (side, side), background)
    x = (side - w) // 2
    y = (side - h) // 2
    canvas.paste(img, (x, y), img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input image path (png/jpg)")
    ap.add_argument("output", help="output image path (png)")
    ap.add_argument("--padding", type=int, default=64, help="pixels of padding on each side")
    args = ap.parse_args()
    pad_square(args.input, args.output, args.padding)
