# scripts/seed_from_id.py
import hashlib
import os
from typing import Optional

import yaml

OVERRIDES_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "seed_overrides.yaml")

def _load_overrides():
    try:
        with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def seed_from_id(card_id: str, extra: Optional[str] = None) -> int:
    overrides = _load_overrides()
    if card_id in overrides:
        return int(overrides[card_id]) & 0xFFFFFFFF
    norm = f"{card_id}".lower().strip()  # ← no subject salt
    digest = hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


if __name__ == "__main__":
    import sys
    print(seed_from_id(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
