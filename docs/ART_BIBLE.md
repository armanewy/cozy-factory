Cozy Sticker V1 — Locked Style

This spec codifies the look used by the generator scripts and ComfyUI workflow and makes outputs reproducible.

Core prompt (objects/buildings)
- cozy sticker icon, single inanimate prop object, front 3/4 view, chunky hand‑inked line art, flat cel shading, soft ambient occlusion, smooth rounded shapes, pastel palette, minimal detail, clean silhouette, centered, product‑shot composition, no text, no background

Negative (objects/buildings)
- photo, photorealistic, painterly texture, gritty, noisy, grainy, text, watermark, logo, busy scene, background, multiple objects, duplicates, people, animals, harsh shadows, glare

Sampler & parameters
- model: `stabilityai/stable-diffusion-xl-base-1.0`
- scheduler/sampler: DPMSolverMultistep (Diffusers) / dpmpp_2m karras (Comfy)
- steps: 28
- guidance: 6.5
- size: 1024×1024
- seed: `seed_from_id(card_id)` (deterministic per id). Saved to `assets/meta/<id>.json` and reused unless `--reseed`.

Post‑process (programmatic, not prompt)
- matte bleed: 2 px (avoid edge halos)
- silhouette outline: white, 28 px, alpha 255
- padding: 64 px on all sides, transparent canvas
- The generator pre‑pads, draws the outline, checks edges, and re‑strokes with extra pad if the outline touches the canvas.

Profiles & cutout
- cozy_sticker_v1 (default): objects/buildings and vegetation; segmentation cutout.
- cozy_sticker_char_v1: animals/mascots; segmentation cutout.
  - Note: crops/terrain use the default style plus per‑card negatives to ban sky/landscape/patterns instead of a separate nature style.

Subject strings (what varies per card)
- Keep short and literal; style words are centralized in the generator.
- Examples: compact bakery oven; small wooden grain mill with a water wheel; vintage street lamp on a tiny round cobblestone base.

Usage (Diffusers script)
- Single card:
  `python scripts/generate_card.py --subject "a compact bakery oven with a tray of warm bread, tiny chimney puff" --id bakery_001 --style cozy_sticker_v1`
- Batch from cards.json (uses `art_subject`):
  `python scripts/generate_all_cards.py --style cozy_sticker_v1`

Optional style lock booster
- You can pass a style LoRA: `--lora path/to/cozy_style.safetensors`. The path is recorded in metadata for reproducibility.

ComfyUI parity (optional)
- Use sampler `dpmpp_2m` + scheduler `karras`, 28 steps, cfg 6.5, SDXL base. Extract alpha via Rembg (or color key), then add the white sticker outline in post.

QA & seed search (recommended)
- `scripts/auto_generate_cards.py` creates N candidates per card with unique seeds, scores them with CLIP, and picks the best subject-matching seed. Profiles (object/building/character) select class-specific negatives; vegetation-specific negatives are provided per card.

Principles
- Style consistency lives in one place (this file + generator defaults). Card JSON only provides the subject and optional per‑card negatives.
- The white sticker border is always programmatic for pixel‑perfect consistency.
- Seeds are card‑id based and preserved so art is repeatable even if the prompt is tweaked.
