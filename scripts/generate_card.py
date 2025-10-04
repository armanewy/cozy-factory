import argparse, json, os, platform, time
from io import BytesIO
from statistics import median
import sys
from pathlib import Path

from PIL import Image, ImageChops
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

from seed_from_id import seed_from_id
from pad_square import pad_square
from rembg import remove as rembg_remove
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.art_post.stroke_and_bleed import apply_stroke_and_bleed

STYLES = {
    # New locked-in style matching assets/art/cards/bakery_002.png
    # We render a single cozy prop with chunky line art and then add a
    # programmatic white sticker outline in post for pixel-perfect consistency.
    "cozy_sticker_v1": {
        "prelude": (
            "cozy sticker icon, single inanimate prop object, front 3/4 view, chunky hand-inked line art, "
            "flat cel shading, soft ambient occlusion, smooth rounded shapes, pastel palette, "
            "minimal detail, clean silhouette, centered, product-shot composition, no text, no background"
        ),
        "negative": (
            "photo, photorealistic, painterly texture, gritty, noisy, grainy, text, watermark, logo, "
            "busy scene, background, multiple objects, duplicates, people, harsh shadows, glare"
        ),
        # highest-priority negatives that must be kept even if we trim
        "priority_negative": "no animals, animal, mascot, character, creature, plush, toy, face, person, human",
        # optional priority positive keywords
        "priority_positive": "technical prop, device, object",
        "steps": 28,
        "guidance": 6.5,
        # Post-process sticker outline (around silhouette)
        "stroke_rgb": (255, 255, 255),
        "stroke_alpha": 255,
        "stroke_px": 28,
        "bleed": 2,
        # Optional LoRA trigger token; used if --lora is passed
        "lora_token": "cozyStickerV1Style",
    },
    # Character/animal variant for cards that should be creatures (e.g., cow)
    "cozy_sticker_char_v1": {
        "prelude": (
            "cozy sticker icon, cute mascot character, front 3/4 view, chunky hand-inked line art, "
            "flat cel shading, soft ambient occlusion, smooth rounded shapes, pastel palette, minimal detail, clean silhouette, centered"
        ),
        "negative": (
            "photo, photorealistic, painterly texture, gritty, noisy, grainy, text, watermark, logo, busy scene, background, duplicates, people, harsh shadows, glare"
        ),
        "steps": 28,
        "guidance": 6.5,
        "stroke_rgb": (255, 255, 255),
        "stroke_alpha": 255,
        "stroke_px": 28,
        "bleed": 2,
        "lora_token": "cozyStickerV1Style",
    },
    # keep your old look available if you want to compare side-by-side
    "cartoon_sticker": {
        "prelude": "cartoon, cel-shaded, flat colors, bold outline, sticker, isometric, single object, centered, large scale, plain white background, clean silhouette, no scene",
        "negative": "photo, realistic, painterly, textured, grainy, noisy, blurry, text, watermark, logo, dark background, black background, busy scene, multiple, pair, duplicate, extra object",
        "steps": 28,
        "guidance": 6.5,
    },
    "cozy_diorama": {
        "prelude": "stylized diorama, hand-painted miniature, soft edges, pastel palette",
        "negative": "photo, realistic, harsh texture, heavy noise, text, watermark, logo",
        "steps": 30,
        "guidance": 7.0,
    },
}
DEFAULT_STYLE = "cozy_sticker_v1"
BUILD_MANIFEST = "assets/meta/build_manifest.json"


def clip_trim(pipe, text: str, budget: int = 70) -> str:
    """Trim comma-separated prompt to fit SDXL's CLIP token budget (77)."""
    tok1 = pipe.tokenizer
    tok2 = getattr(pipe, "tokenizer_2", tok1)

    def n_tokens(s: str) -> int:
        return max(
            len(tok1(s, return_tensors="pt").input_ids[0]),
            len(tok2(s, return_tensors="pt").input_ids[0]),
        )

    if n_tokens(text) <= budget:
        return text

    parts = [p.strip() for p in text.split(",")]
    out = []
    for p in parts:
        trial = ", ".join(out + [p])
        if n_tokens(trial) <= budget:
            out.append(p)
        else:
            break
    return ", ".join(out)


def _write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _ensure_dirs():
    os.makedirs("assets/art/cards", exist_ok=True)
    os.makedirs("assets/meta", exist_ok=True)
    os.makedirs("temp", exist_ok=True)


def render_sdxl(prompt, negative, seed, steps, guidance, width, height, model_id, lora_path: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        use_safetensors=True,
    )
    # Optional LoRA for style locking
    if lora_path:
        try:
            lora_input = lora_path
            if os.path.isfile(lora_input):
                lora_input = os.path.dirname(lora_input) or "."
            pipe.load_lora_weights(lora_input)
        except Exception as e:
            print(f"[warn] failed to load LoRA '{lora_path}': {e}")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Robust device/dtype move across Diffusers versions
    try:
        pipe.to(device=device, dtype=dtype)  # newer Diffusers
    except TypeError:
        pipe.to(device)  # older Diffusers
        for m in [pipe.unet, getattr(pipe, "text_encoder", None), getattr(pipe, "text_encoder_2", None), pipe.vae]:
            if m is not None:
                m.to(dtype=dtype)

    # Trim prompts to avoid CLIP-length chatter
    prompt_t = clip_trim(pipe, prompt, budget=75)
    negative_t = clip_trim(pipe, negative or "", budget=75)

    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt_t,
        negative_prompt=negative_t,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    return image, prompt_t, negative_t


def cutout_by_bgcolor(img: Image.Image, tol: int = 10):
    """
    Create alpha by removing pixels close to the detected background color.
    Background color is estimated from the four corners (median).
    Returns (rgba_image, bg_rgb_tuple).
    """
    rgb = img.convert("RGB")
    W, H = rgb.size
    s = max(8, min(W, H) // 16)  # corner sample size
    corners = [
        rgb.crop((0, 0, s, s)),
        rgb.crop((W - s, 0, W, s)),
        rgb.crop((0, H - s, s, H)),
        rgb.crop((W - s, H - s, W, H)),
    ]
    pixels = []
    for c in corners:
        pixels += list(c.getdata())
    r = int(median(p[0] for p in pixels))
    g = int(median(p[1] for p in pixels))
    b = int(median(p[2] for p in pixels))
    bg = (r, g, b)

    bg_img = Image.new("RGB", rgb.size, bg)
    diff = ImageChops.difference(rgb, bg_img).convert("L")  # per-pixel distance from bg color
    # Threshold: keep where difference > tol
    mask = diff.point(lambda v: 255 if v > tol else 0, "L")

    rgba = img.convert("RGBA")
    rgba.putalpha(mask)
    return rgba, bg


def remove_bg_rgba(img: Image.Image) -> Image.Image:
    """Alpha-matte segmentation tuned for the cozy sticker pipeline."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    out = rembg_remove(
        buf.getvalue(),
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
    )
    return Image.open(BytesIO(out)).convert("RGBA")


def run_generation(
    subject: str,
    cid: str,
    negative: str,
    steps: int,
    guidance: float,
    padding: int,
    width: int,
    height: int,
    model_id: str,
    lora_path: str | None,
    add_frame: bool,
    style: str,
    cutout_mode: str,
    seed_arg: int | None,
    reseed: bool = False,
):
    _ensure_dirs()
    meta_path = Path("assets") / "meta" / f"{cid}.json"
    seed = seed_arg

    # Prefer previous saved seed if present (stable across prompt edits)
    if seed is None and not reseed:
        prev = _read_json(str(meta_path))
        if prev and "seed" in prev:
            try:
                seed = int(prev["seed"])
            except Exception:
                seed = None

    # Fallback to ID-only seed (so prompts can change without shifting the seed)
    if seed is None:
        seed = seed_from_id(cid)

    style_cfg = STYLES[style]

    # Defaults from style if user didn't set
    if steps is None:
        steps = style_cfg["steps"]
    if guidance is None:
        guidance = style_cfg["guidance"]
    if width is None:
        width = style_cfg.get("width", width)
    if height is None:
        height = style_cfg.get("height", height)

    # Compose prompts
    prompt_parts = [style_cfg.get("priority_positive", ""), style_cfg["prelude"], subject.strip()]
    # If a style LoRA is provided and this style defines a trigger token, include it
    if lora_path and style_cfg.get("lora_token"):
        prompt_parts.insert(0, style_cfg["lora_token"])  # put token first for stronger influence
    full_prompt = ", ".join(part for part in prompt_parts if part)
    # Priority negatives first to ensure they survive CLIP trimming
    neg_parts = [style_cfg.get("priority_negative", ""), style_cfg["negative"], (negative or "")]
    full_negative = ", ".join([p for p in neg_parts if p]).strip(", ")

    raw, final_prompt, final_negative = render_sdxl(
        full_prompt, full_negative, seed, steps, guidance, width, height, model_id, lora_path
    )

    # Generate alpha; default to rembg with tuned alpha-matting
    if cutout_mode == "auto":
        cutout, bg_used = cutout_by_bgcolor(raw, tol=10)
    else:
        cutout = remove_bg_rgba(raw)
        bg_used = None

    # Matte bleed + cozy stroke for consistent sticker silhouette
    # Style-specific sticker outline and matte bleed
    cutout = apply_stroke_and_bleed(
        cutout,
        bleed_radius=int(style_cfg.get("bleed", 2)),
        stroke_px=int(style_cfg.get("stroke_px", 3)),
        stroke_rgb=tuple(style_cfg.get("stroke_rgb", (42, 36, 32))),
        stroke_alpha=int(style_cfg.get("stroke_alpha", 180)),
    )

    # Pad to square (transparent) and optionally frame
    tmp_cut = os.path.join("temp", f"{cid}_no_bg.png")
    cutout.save(tmp_cut)

    padded_path = os.path.join("temp", f"{cid}_padded.png")
    pad_square(tmp_cut, padded_path, padding=padding)

    final_path = os.path.join("assets", "art", "cards", f"{cid}.png")
    if add_frame:
        from frame_card import frame_card
        frame_card(padded_path, final_path)
    else:
        os.replace(padded_path, final_path)

    # Per-card meta
    card_meta = {
        "id": cid,
        "style": style,
        "subject": subject,
        "prompt_final": final_prompt,
        "negative_final": final_negative,
        "seed": seed,
        "model_id": model_id,
        "scheduler": "DPMSolverMultistepScheduler",
        "steps": steps,
        "guidance": guidance,
        "width": int(width),
        "height": int(height),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "diffusers": __import__("diffusers").__version__,
        "platform": platform.platform(),
        "time_unix": int(time.time()),
        "cutout_mode": cutout_mode,
        "bg_color_detected": bg_used,
    }
    if lora_path:
        card_meta["lora"] = lora_path
    if "refiner_strength" in style_cfg:
        card_meta["refiner_strength"] = style_cfg["refiner_strength"]

    _write_json(os.path.join("assets", "meta", f"{cid}.json"), card_meta)

    manifest = _read_json(BUILD_MANIFEST) or {"cards": {}}
    manifest["cards"][cid] = card_meta
    _write_json(BUILD_MANIFEST, manifest)

    print(f"[ok] {final_path}")
    return final_path


def main():
    ap = argparse.ArgumentParser(description="subject + id -> assets/art/cards/<id>.png reproducibly")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--negative", default="low quality, deformed, extra fingers, blurry, text, watermark, logo")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--padding", type=int, default=64)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--lora", default=None, help="optional LoRA weights path for style locking")
    ap.add_argument("--frame", action="store_true", help="add subtle frame/shadow")
    ap.add_argument("--style", default=DEFAULT_STYLE, choices=list(STYLES.keys()))
    ap.add_argument("--cutout", default="rembg", choices=["auto", "rembg"], help="auto=color-key; rembg=segment")
    ap.add_argument("--seed", type=int, default=None, help="explicit seed (overrides everything)")
    ap.add_argument("--reseed", action="store_true", help="ignore previous seed and recompute")
    args = ap.parse_args()

    run_generation(
        subject=args.subject,
        cid=args.id,
        negative=args.negative,
        steps=args.steps,
        guidance=args.guidance,
        padding=args.padding,
        width=args.width,
        height=args.height,
        model_id=args.model,
        lora_path=args.lora,
        add_frame=args.frame,
        style=args.style,
        cutout_mode=args.cutout,
        seed_arg=args.seed,
        reseed=args.reseed
    )


if __name__ == "__main__":
    main()
