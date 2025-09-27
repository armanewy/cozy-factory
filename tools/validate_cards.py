import json, sys
from jsonschema import validate, Draft7Validator

def main(cards_path: str, schema_path: str):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    with open(cards_path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    Draft7Validator.check_schema(schema)
    if not isinstance(cards, list):
        raise SystemExit("cards.json must be a list of card objects")

    for i, card in enumerate(cards):
        try:
            validate(instance=card, schema=schema)
        except Exception as e:
            raise SystemExit(f"❌ Card index {i} ({card.get('id')}): {e}")
    print(f"✅ {len(cards)} cards validated")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/validate_cards.py <cards.json> <schema.json>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
