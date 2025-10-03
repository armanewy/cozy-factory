import json, os, shutil
from pathlib import Path

CARDS = Path("assets/meta/cards.json")
ART_DIR = Path("assets/art/cards")
DATA_ROOT = Path("data/lora/cozy_sticker_v1")

TARGET_IDS = [
    "wheat_field_001",
    "mill_001",
    "bakery_001",
    "water_pump_001",
    "farm_001",
    "cow_001",
    "dairy_001",
    "sugarcane_001",
    "sugar_mill_001",
    "pastry_001",
]

TEMPLATE = (
    "cozyStickerV1Style, cozy kawaii sticker, single prop, front 3/4 view, "
    "chunky hand-inked line art, flat cel shading, soft ambient occlusion, pastel palette, "
    "minimal detail, clean silhouette, centered — {subject}"
)


def main():
    cards = json.loads(CARDS.read_text("utf-8"))
    by_id = {c["id"]: c for c in cards}

    images_dir = DATA_ROOT / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions = {}
    for cid in TARGET_IDS:
        c = by_id[cid]
        art_id = c["art_id"]
        src = ART_DIR / f"{art_id}.png"
        dst = images_dir / f"{art_id}.png"
        if not src.exists():
            print(f"[warn] missing art for {cid} ({art_id})")
            continue
        shutil.copy2(src, dst)
        subject = c.get("art_subject") or c.get("art_prompt") or c.get("name")
        captions[dst.name] = TEMPLATE.format(subject=subject)

    (DATA_ROOT / "captions.json").write_text(json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] dataset prepared at {DATA_ROOT} with {len(captions)} items")


if __name__ == "__main__":
    main()

