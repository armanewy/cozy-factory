import argparse, json, os
from pathlib import Path

TEMPLATE = (
    "cozyStickerV1Style, cozy kawaii sticker, single prop, front 3/4 view, "
    "chunky hand-inked line art, flat cel shading, soft ambient occlusion, pastel palette, "
    "minimal detail, clean silhouette, centered — {subject}"
)


def scaffold(root: Path) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    captions_path = root / "captions.json"
    if not captions_path.exists():
        captions_path.write_text("{}\n", encoding="utf-8")
    print(f"[ok] dataset root: {root}")


def build_captions(root: Path, default_subject: str | None) -> None:
    images = sorted((root / "images").glob("*.png"))
    captions_path = root / "captions.json"
    existing = {}
    if captions_path.exists():
        try:
            existing = json.loads(captions_path.read_text("utf-8"))
        except Exception:
            existing = {}

    for img in images:
        key = img.name
        if key not in existing:
            subject = (default_subject or img.stem.replace("_", " ")).strip()
            existing[key] = TEMPLATE.format(subject=subject)

    captions_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] wrote {captions_path} for {len(images)} images")


def main():
    ap = argparse.ArgumentParser(description="Scaffold/prepare LoRA dataset with caption JSON")
    ap.add_argument("--root", required=True, help="dataset root (will contain images/ and captions.json)")
    ap.add_argument("--default-subject", default=None, help="fallback subject text if filename is not descriptive")
    args = ap.parse_args()

    root = Path(args.root)
    scaffold(root)
    build_captions(root, args.default_subject)


if __name__ == "__main__":
    main()

