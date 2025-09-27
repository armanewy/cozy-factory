from PIL import Image, ImageFilter, ImageOps
import os, argparse

def frame_card(input_path, output_path, border=12, shadow=18, bg=(0,0,0,0)):
    img = Image.open(input_path).convert("RGBA")
    # Drop shadow
    shadow_layer = Image.new("RGBA", (img.width+shadow*2, img.height+shadow*2), (0,0,0,0))
    blurred = Image.new("RGBA", shadow_layer.size, (0,0,0,0))
    offset = (shadow, shadow)
    blurred.paste(img, (shadow, shadow), img)
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=shadow//2))
    # Canvas + center image
    canvas = Image.new("RGBA", shadow_layer.size, bg)
    canvas = Image.alpha_composite(canvas, blurred)
    canvas.paste(img, offset, img)
    # Border
    framed = ImageOps.expand(canvas, border=border, fill=(255,255,255,20))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    framed.save(output_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--border", type=int, default=12)
    ap.add_argument("--shadow", type=int, default=18)
    args = ap.parse_args()
    frame_card(args.input, args.output, args.border, args.shadow)
