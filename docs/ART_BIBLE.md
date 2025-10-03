Cozy Sticker V1 — Locked Style

This spec locks the art look to match `assets/art/cards/bakery_002.png` and makes it reproducible across cards and rebuilds.

Prompt (positive)
- cozy kawaii sticker, single prop, front 3/4 view, chunky hand-inked line art, flat cel shading, soft ambient occlusion, smooth rounded shapes, pastel palette, minimal detail, clean silhouette, centered, product-shot composition, no text, no background

Negative
- photo, photorealistic, painterly texture, gritty, noisy, grainy, text, watermark, logo, busy scene, background, multiple objects, duplicates, people, animals, harsh shadows, glare

Sampler & Parameters
- model: `stabilityai/stable-diffusion-xl-base-1.0`
- sampler/scheduler: DPMSolverMultistep (diffusers) / dpmpp_2m karras (Comfy)
- steps: 28
- guidance: 6.5
- size: 1024×1024
- seed: `seed_from_id(card_id)` (deterministic per id). Saved to `assets/meta/<id>.json` and reused unless `--reseed`.

Post-process (programmatic, not prompt)
- matte bleed: 2 px (avoid edge halos)
- silhouette outline: white, 28 px, alpha 255 (sticker look)
- padding: 64 px on all sides, transparent canvas

Subject Strings (what varies per card)
- Keep short and literal. Examples:
  - wheat_field_001: a rectangular patch of golden wheat in neat rows on a tiny soil tile, a few stones and tufts
  - mill_001: a small wooden grain mill with a water wheel and stone base, warm window glow
  - bakery_001: a compact bakery oven with a tray of warm bread, tiny chimney puff
  - market_001: a small market stall with a striped awning and a few crates of goods
  - lamp_post_001: a vintage street lamp on a tiny round cobblestone base, glowing glass lantern
  - signboard_001: a wooden hanging sign with carved trim on a metal bracket

Usage (diffusers script)
- Generate one card:
  `python scripts/generate_card.py --subject "a compact bakery oven with a tray of warm bread, tiny chimney puff" --id bakery_001 --style cozy_sticker_v1`

- Batch from `assets/meta/cards.json` (uses `art_subject`):
  `python scripts/generate_all_cards.py --style cozy_sticker_v1`

Optional Style Lock Booster
- You can pass a style LoRA: `--lora path/to/cozy_style.safetensors`. The path is recorded in metadata for reproducibility.

ComfyUI Parity (optional)
- Use sampler `dpmpp_2m` + scheduler `karras`, 28 steps, cfg 6.5, SDXL base. Positive/negative prompts above. Use an Rembg node for alpha and add the white sticker outline in post (not in the prompt).

Principles
- Style words live in one place (this file + generator defaults). Card JSON only supplies the subject.
- The white sticker border is always programmatic for pixel-perfect consistency.
- Seeds are card-id based and preserved so art is repeatable even if the prompt is tweaked.

