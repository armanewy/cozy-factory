Cozy Factory — Developer Guide

This repository contains a small content pipeline for generating cozy, sticker‑style card art and metadata for a deterministic factory‑puzzle prototype. It also includes a few ComfyUI workflows and utilities for validating and regenerating art deterministically.

Repository layout

- Art + metadata
  - assets/art/cards/…: final PNGs used by the game (sticker look, white outline)
  - assets/meta/…: per‑card JSON metadata (seed, prompts, model params, etc.)
  - assets/meta/cards.json: source list of cards (ids, subjects, negatives, tags)
  - assets/meta/build_manifest.json: manifest aggregated during generation

- Generation scripts
  - scripts/generate_card.py: single‑card generator (Diffusers SDXL)
  - scripts/generate_all_cards.py: batch generator over assets/meta/cards.json
  - scripts/auto_generate_cards.py: batch generator with CLIP‑based QA & seed search
  - scripts/frame_card.py, scripts/pad_square.py: post‑processing helpers
  - scripts/seed_from_id.py: stable 32‑bit seeds derived from ids (with overrides)

- Art post‑processing
  - tools/art_post/stroke_and_bleed.py: applies matte‑bleed and white outline

- ComfyUI workflows
  - comfy/workflows/cozy_sticker_v1.json: SDXL base node graph that mirrors the defaults used by the Python scripts
  - comfy/workflows/cozy_diorama.json: alternate workflow used earlier in exploration

- Docs
  - docs/ART_BIBLE.md: style guide (prompt template, palette, outline, output specs)
  - docs/LoRA_TRAINING.md: recipe for training/using a style LoRA (optional)

Environment & setup

- Python 3.12 and a GPU with CUDA are recommended (tested with an RTX 4070).
- Known‑good package set for SDXL in this repo:
  - torch 2.6.0+cu124, torchvision 0.21.0+cu124
  - diffusers 0.35.1, transformers 4.46.3, accelerate, rembg

Create/activate a venv and install:

- python -m venv .venv
- .venv\Scripts\python.exe -m pip install --upgrade pip
- .venv\Scripts\python.exe -m pip install "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" --index-url https://download.pytorch.org/whl/cu124
- .venv\Scripts\python.exe -m pip install diffusers==0.35.1 transformers==4.46.3 accelerate safetensors rembg pillow

Style and determinism

- Styles are defined in scripts/generate_card.py: cozy_sticker_v1 (default), cozy_sticker_nature_v1, cozy_sticker_char_v1. Each style bundles:
  - A positive “prelude” (style adjectives)
  - Priority negatives (e.g., “animal, face” for objects; “machine, screen” for nature)
  - SDXL params (steps, cfg)
  - Sticker outline parameters (stroke thickness/color and matte‑bleed)
  - Default cutout mode: “rembg” segmentation for objects/characters; “auto” color‑key is available and can be chosen per card

- Seeds are derived by scripts/seed_from_id.py from the card id. An optional config/seed_overrides.yaml lets you pin a specific integer for any id.

Core data model

- assets/meta/cards.json holds content used by the batch generators. A card entry:
  - id: unique id (also used for art_id)
  - name
  - tags: e.g., ["processing", "building"], ["farm", "resource"], ["animal"]
  - art_id: file stem in assets/art/cards/<art_id>.png
  - style: optional override (cozy_sticker_v1 default; use nature/char variants when needed)
  - art_subject: literal subject text (no style words)
  - negative: per‑card negatives to suppress undesired features
  - produces/consumes/cost: game data (not used by art pipeline)

Single‑card generation

Examples:

- .venv\Scripts\python.exe scripts\generate_card.py --subject "a compact bakery oven with a tray of warm bread, tiny chimney puff" --id bakery_001 --style cozy_sticker_v1

Useful flags:

- --negative: additional negatives; merged with style and per‑card ones
- --steps / --guidance: SDXL params (styles set defaults when not provided)
- --cutout: style | auto | rembg (alpha extraction)
- --seed: explicit integer to override the id‑based seed
- --frame: add a subtle drop‑shadow frame (for previews)
- --out: override output path (no metadata written if --no-meta also set)

Batch generation (no QA)

- .venv\Scripts\python.exe scripts\generate_all_cards.py --style cozy_sticker_v1
- Flags: --force (regenerate even if PNG exists), --continue (keep going on errors), --style, --lora, --steps, --guidance

Batch generation with QA + seed search (recommended)

- .venv\Scripts\python.exe scripts\auto_generate_cards.py --style cozy_sticker_v1 --attempts 8 --force
- What it does:
  - Loads cards.json
  - For each card, generates N candidates with unique seeds derived from id
  - Scores with CLIP (subject similarity vs class‑specific negative prompts)
  - Picks the best candidate seed and renders the final image with metadata
- Profiles (class‑aware negatives) are chosen per card by style and tags:
  - object: blocks faces/mascots and UI panels
  - building: blocks appliance/screen clutter to keep “toy‑like” shapes
  - nature: blocks machine/screen and any “pattern/wallpaper/landscape/sky” drift
  - character: for animals/mascots only (e.g., cow)
- Subset reruns: add --only id1,id2

Reproducibility & metadata

- Each generated card writes assets/meta/<id>.json with:
  - style, subject, seed, model_id, steps, cfg, width/height
  - final prompts (after token trimming)
  - cutout mode and detected bg color (if any)
  - device/torch/diffusers version snapshot
  - Build manifest is also updated: assets/meta/build_manifest.json

Post‑processing (sticker look)

- tools/art_post/stroke_and_bleed.py:
  - alpha_bleed: fills transparent pixels with a blurred matte to avoid edge halos in atlases
  - outer_stroke + matte: white outline + subtle stroke color (configurable per style)
  - The generator pre‑pads the image before stroking, and adds a final pad after; it also retries with extra padding if the stroked alpha touches the canvas edges.

ComfyUI parity

- comfy/workflows/cozy_sticker_v1.json mirrors defaults used in the Python scripts (SDXL base, dpmpp_2m + karras, 28 steps, cfg ~6.5). You can plug the same subject, seed, steps/cfg and export a PNG manually. Keep the cutout + sticker outline steps in the Python pipeline for consistent borders.

LoRA (optional)

- docs/LoRA_TRAINING.md provides a minimal training script + dataset scaffolding. If you use a LoRA, pass --lora to the generators; style tokens are appended automatically when present.

Troubleshooting

- Tokenizer warnings (77‑token budget): scripts/generate_card.py trims both positive/negative prompts to fit both SDXL text encoders.
- UTF‑8 BOM errors on Windows: some editors save JSON with a BOM. If you see "Unexpected UTF‑8 BOM", resave the file as UTF‑8 (no BOM) or rewrite it from PowerShell using Set‑Content -Encoding UTF8.
- Borders clipped: the generator pre‑pads, strokes, checks edges, and re‑strokes with extra pad if needed. If you still see clipping on a specific id, increase padding with --padding in generate_card.py or add a style override.
- Object drift into “appliance” look: reinforce per‑card negatives in assets/meta/cards.json and rerun with --attempts 8+ via auto_generate_cards.py.
- Nature drifting into “wallpaper/landscape”: the nature profile blocks these; ensure the card has style cozy_sticker_nature_v1 or tags that trigger that profile.

Notes & status

- The Wheat Field card has been a deliberate hotspot. If you need to regenerate it, prefer running scripts/auto_generate_cards.py with --only wheat_field_001 and a higher --attempts to find a clean seed. If an SDXL render still produces interior artifacts, consider a manual PNG edit or an inpaint pass in ComfyUI; the pipeline will preserve the sticker outline and padding.

Operational tips

- Keep subjects literal and short in cards.json; all style words live in the styles table.
- For buildings, choose the “toy‑like” silhouettes (few large rounded shapes). Ban gauges, UI panels, bolts, and wires in negatives for extra simplicity.
- For signs or anything with text, keep the art blank and render UI text in‑engine.

License & credits

- This repository is for internal prototyping. Do not add third‑party assets without verifying their licenses. The SDXL model license applies when using its outputs. No additional license metadata is included in this repo.

