What is a LoRA?
- A tiny add-on that “nudges” a big model toward a specific look. It’s a few megabytes of learned deltas, not a whole model. Loading it during generation locks line weight, shading, and palette without changing your prompts.

Recommended plan for Cozy Sticker V1
1) Curate 30–50 reference icons that match the bakery_002 look. Prefer vector/flat-sticker props with chunky outlines and minimal detail. PNG with transparency, 768–1024 px.
2) Put them in `data/lora/cozy_sticker_v1/images`. Name files like `oven.png`, `mill.png`, etc.
3) Create a simple `captions.json` mapping filename → subject string (short and literal):
   {"oven.png": "a compact bakery oven", "mill.png": "a small wooden grain mill"}
   Or run the helper script to scaffold captions: `python scripts/prepare_lora_dataset.py --root data/lora/cozy_sticker_v1`.
4) Train the LoRA:
   accelerate launch scripts/train_lora_sdxl.py \
     --data_root data/lora/cozy_sticker_v1 \
     --output models/lora/cozy_sticker_v1.safetensors \
     --train_steps 3500 --batch 2 --lr 1e-4 --rank 16

Parameters
- Base: SDXL base 1.0
- Resolution: 1024
- Steps: 3–5k (watch loss/overfit)
- Rank: 8–16 (start 16)
- LR: 1e-4 (UNet/TextEncoder), cosine decay
- Caption template used internally: `cozyStickerV1Style, cozy kawaii sticker, single prop, front 3/4 view, chunky hand-inked line art, flat cel shading, pastel palette, minimal detail, clean silhouette, centered — {subject}`
  - `cozyStickerV1Style` is the trigger token. The generator automatically adds it when a LoRA path is supplied.

Using the LoRA
- Single card: `python scripts/generate_card.py --subject "a compact bakery oven with a tray of warm bread, tiny chimney puff" --id bakery_001 --style cozy_sticker_v1 --lora models/lora/cozy_sticker_v1.safetensors`
- Batch: `python scripts/generate_all_cards.py --style cozy_sticker_v1 --lora models/lora/cozy_sticker_v1.safetensors`

Quality tips
- Keep subjects centered and single-prop. Remove busy backgrounds before training.
- If outputs skew realistic, raise CFG a bit (7.0) or add more flat, high-contrast references.
- If faces/eyes creep in, add them to negatives and remove people references entirely.

