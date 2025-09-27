# Phase 1 - Cozy Bakery Vertical Slice

## What you get
- Deterministic art for 6 starter cards
- Per-card metadata + build manifest
- Batch regen
- JSON schema + validator
- Local gallery
- 60s sim (wheat -> flour -> bread -> gold)
- Matte bleed + cozy stroke pass for card art

## Commands
```powershell
# Validate + batch generate missing art
python tools/validate_cards.py assets/meta/cards.json tools/card_schema.json
python scripts/generate_all_cards.py
# (rebuild everything)
python scripts/generate_all_cards.py --force

# Run sim
python tools/sim.py --ticks 60

# Open gallery
python -m http.server -d tools 8000
# open http://localhost:8000/gallery.html
```

## Art polish
```powershell
# Add matte bleed + outer stroke (in-place)
python tools/art_post/stroke_and_bleed.py assets/art/cards/bakery.png

# Or write to a new path
python tools/art_post/stroke_and_bleed.py assets/raw/bakery.png temp/bakery_polished.png
```
