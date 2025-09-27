import os, json, argparse

CARDS = [
  {
    "id":"wheat_field_001","name":"Wheat Field","rarity":"common","tags":["farm","resource"],
    "art_id":"wheat_field_001",
    "art_prompt":"cozy golden wheat field, warm dusk light, soft bokeh, painterly, gentle vignette, inviting, isometric angle",
    "negative":"low quality, deformed, text, watermark, logo, frame",
    "produces":[{"resource":"wheat","rate":2}],
    "consumes":[],
    "cost":{"wood":5}
  },
  {
    "id":"mill_001","name":"Mill","rarity":"common","tags":["processing","building"],
    "art_id":"mill_001",
    "art_prompt":"cozy wooden mill with turning wheel, lantern glow, stone base, warm lamplight, soft bokeh, painterly, isometric",
    "negative":"low quality, deformed, text, watermark, logo, frame",
    "produces":[{"resource":"flour","rate":1}],
    "consumes":[{"resource":"wheat","rate":2}],
    "cost":{"wood":8,"stone":4}
  },
  {
    "id":"bakery_001","name":"Bakery","rarity":"uncommon","tags":["processing","building"],
    "art_id":"bakery_001",
    "art_prompt":"cozy bakery storefront, warm lamplight, fresh bread in window, soft bokeh, painterly, dusk, isometric",
    "negative":"low quality, deformed, text, watermark, logo, frame",
    "produces":[{"resource":"bread","rate":1}],
    "consumes":[{"resource":"flour","rate":1}],
    "cost":{"wood":10,"stone":6}
  },
  {
    "id":"market_001","name":"Market Stall","rarity":"common","tags":["trade","building"],
    "art_id":"market_001",
    "art_prompt":"cozy market stall with striped awning, baskets of goods, warm lanterns, painterly, soft bokeh, isometric",
    "negative":"low quality, deformed, text, watermark, logo, frame",
    "produces":[{"resource":"gold","rate":1}],
    "consumes":[{"resource":"bread","rate":1}],
    "cost":{"wood":6}
  },
  {
    "id":"lamp_post_001","name":"Lamp Post","rarity":"common","tags":["decor"],
    "art_id":"lamp_post_001",
    "art_prompt":"cozy iron lamp post with warm glow, cobblestone base, soft fog, painterly, soft bokeh, isometric",
    "negative":"low quality, deformed, text, watermark, logo, frame",
    "produces":[],
    "consumes":[],
    "cost":{"wood":1,"iron":1}
  },
  {
    "id":"signboard_001","name":"Signboard","rarity":"common","tags":["decor"],
    "art_id":"signboard_001",
    "art_prompt":"cozy wooden hanging signboard, carved trim, warm lamplight nearby, painterly, soft bokeh, isometric",
    "negative":"low quality, deformed, text, watermark, logo, frame",
    "produces":[],
    "consumes":[],
    "cost":{"wood":2}
  }
]

SCHEMA = {
  "$schema":"http://json-schema.org/draft-07/schema#",
  "title":"Cozy Card","type":"object",
  "required":["id","name","rarity","tags","art_id","art_prompt"],
  "properties":{
    "id":{"type":"string","pattern":"^[a-z0-9_]+$"},
    "name":{"type":"string","minLength":1},
    "rarity":{"type":"string","enum":["common","uncommon","rare","epic","legendary"]},
    "tags":{"type":"array","items":{"type":"string"}},
    "art_id":{"type":"string","pattern":"^[a-z0-9_]+$"},
    "art_prompt":{"type":"string","minLength":4},
    "negative":{"type":"string"},
    "produces":{"type":"array","items":{
      "type":"object","required":["resource","rate"],
      "properties":{"resource":{"type":"string"},"rate":{"type":"number","minimum":0}}
    }},
    "consumes":{"type":"array","items":{
      "type":"object","required":["resource","rate"],
      "properties":{"resource":{"type":"string"},"rate":{"type":"number","minimum":0}}
    }},
    "cost":{"type":"object","additionalProperties":{"type":"number","minimum":0}}
  },
  "additionalProperties":False
}

def write_json(path, data, skip_if_exists=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if skip_if_exists and os.path.exists(path):
        print(f"⏭️  exists, skipping {path}")
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"✅ wrote {path}")

def main(force: bool):
    write_json("assets/meta/cards.json", CARDS, skip_if_exists=not force)
    write_json("tools/card_schema.json", SCHEMA, skip_if_exists=not force)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()
    main(args.force)
