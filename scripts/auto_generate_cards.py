import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CARDS = Path("assets/meta/cards.json")
PY = os.environ.get("PY", os.path.join(".venv", "Scripts", "python.exe"))


def seed_stream(base: int, n: int = 12):
    for i in range(n):
        yield (base + i) & 0xFFFFFFFF


class CLIPGuard:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.inference_mode()
    def score(self, image: Image.Image, pos_text: str, neg_texts: list[str]) -> tuple[float, float]:
        inputs_pos = self.proc(text=[pos_text], images=image, return_tensors="pt").to(self.device)
        out_pos = self.model(**inputs_pos)
        pos = out_pos.logits_per_image.flatten()[0].item()
        max_neg = -1e9
        for nt in neg_texts:
            inputs_neg = self.proc(text=[nt], images=image, return_tensors="pt").to(self.device)
            out_neg = self.model(**inputs_neg)
            max_neg = max(max_neg, out_neg.logits_per_image.flatten()[0].item())
        return pos, max_neg


def load_cards():
    return json.loads(CARDS.read_text("utf-8"))


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


PROFILES: dict[str, dict[str, list[str] | str]] = {
    "object": {
        "pos_hint": "technical device, object, prop",
        "neg": [
            "animal, mascot, face, character, plush toy, eyes, cheeks",
            "person, human, face",
        ],
    },
    "nature": {
        "pos_hint": "terrain tile, vegetation, plants, crops",
        "neg": [
            "robot, machine, device, metallic panel, console, monitor, screen, gauge",
            "animal, mascot, face, character, plush toy, eyes, cheeks",
        ],
    },
    "character": {
        "pos_hint": "cute mascot character, animal",
        "neg": [
            "robot, machine, device, console, monitor, screen, gauge",
            "building, factory, architecture",
        ],
    },
}


def profile_for_card(card: dict, style_default: str) -> str:
    style = card.get("style", style_default) or ""
    tags = [t.lower() for t in card.get("tags", [])]
    if "nature" in style:
        return "nature"
    if "char" in style:
        return "character"
    if "animal" in tags:
        return "character"
    if any(t in tags for t in ("farm", "resource")) and "field" in card.get("name", "").lower():
        return "nature"
    return "object"


def main(style_default: str, steps: int, guidance: float, attempts: int, force: bool):
    cards = load_cards()
    qa = CLIPGuard()
    cand_root = Path("temp/candidates")
    accepted = 0
    for card in cards:
        art_id = card["art_id"]
        out_final = Path("assets/art/cards/") / f"{art_id}.png"
        if out_final.exists() and not force:
            print(f"[skip] {art_id} (exists)")
            continue
        subject = card.get("art_subject") or card.get("art_prompt") or card["name"]
        style = card.get("style", style_default)
        neg = card.get("negative", "")

        # Profile-driven QA setup
        profile = profile_for_card(card, style_default)
        prof = PROFILES[profile]
        pos_text = f"{subject}. {prof['pos_hint']}"
        neg_texts = list(prof["neg"])  # type: ignore[index]

        best = None
        best_seed = None
        # Try a small seed window, keep the best margin
        base_seed = int.from_bytes(art_id.encode("utf-8"), "big") & 0xFFFFFFFF
        for seed in seed_stream(base_seed, attempts):
            cand_path = cand_root / f"{art_id}__{seed}.png"
            ensure_dir(cand_path)
            cmd = [
                PY,
                "scripts/generate_card.py",
                "--subject",
                subject,
                "--id",
                art_id,
                "--style",
                style,
                "--steps",
                str(steps),
                "--guidance",
                str(guidance),
                "--seed",
                str(seed),
                "--no-meta",
                "--out",
                str(cand_path),
            ]
            subprocess.check_call(cmd)
            img = Image.open(cand_path).convert("RGB")
            pos, neg_max = qa.score(img, pos_text, neg_texts)
            margin = pos - neg_max
            if best is None or margin > best:
                best = margin
                best_seed = seed

        if best_seed is None:
            raise RuntimeError(f"no candidate chosen for {art_id}")

        # Render the accepted image deterministically and write metadata
        cmd = [
            PY,
            "scripts/generate_card.py",
            "--subject",
            subject,
            "--id",
            art_id,
            "--style",
            style,
            "--steps",
            str(steps),
            "--guidance",
            str(guidance),
            "--seed",
            str(best_seed),
        ]
        subprocess.check_call(cmd)
        accepted += 1
        print(f"[ok] {art_id} profile={profile} seed={best_seed} margin={best:.3f}")

    print(f"[done] accepted={accepted}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-generate all cards with QA (CLIP) and fixed seeds")
    ap.add_argument("--style", default="cozy_sticker_v1")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=6.5)
    ap.add_argument("--attempts", type=int, default=8, help="candidate seeds per card")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(args.style, args.steps, args.guidance, args.attempts, args.force)
