import argparse
import json, os, subprocess, sys

CARDS = "assets/meta/cards.json"
PY = sys.executable or "python"


def art_path(art_id: str) -> str:
    return os.path.join("assets", "art", "cards", f"{art_id}.png")


def main(force: bool, keep_going: bool, style: str | None = None, lora: str | None = None, steps: int | None = None, guidance: float | None = None) -> None:
    with open(CARDS, "r", encoding="utf-8") as f:
        cards = json.load(f)

    made, skipped = 0, 0
    for card in cards:
        art_id = card["art_id"]
        # Prefer a clean subject field if present; fall back to legacy art_prompt
        prompt = card.get("art_subject") or card.get("art_prompt")
        negative = card.get("negative", "")
        output_path = art_path(art_id)

        if not force and os.path.exists(output_path):
            print(f"[skip] {art_id} (exists)")
            skipped += 1
            continue

        cmd = [
            PY,
            "scripts/generate_card.py",
            "--subject",
            prompt,
            "--id",
            art_id,
            "--negative",
            negative,
        ]
        if style:
            cmd += ["--style", style]
        if lora:
            cmd += ["--lora", lora]
        if steps is not None:
            cmd += ["--steps", str(steps)]
        if guidance is not None:
            cmd += ["--guidance", str(guidance)]
        print("[run]", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
            made += 1
        except subprocess.CalledProcessError as exc:
            if keep_going:
                print(f"[warn] {art_id} failed with code {exc.returncode}; continuing")
                continue
            raise

    print(f"[done] generated={made} skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue", dest="keep_going", action="store_true", help="continue on individual card errors")
    parser.add_argument("--style", default="cozy_sticker_v1", help="generation style to use")
    parser.add_argument("--lora", default=None, help="optional LoRA weights path")
    parser.add_argument("--steps", type=int, default=None, help="override steps")
    parser.add_argument("--guidance", type=float, default=None, help="override guidance scale")
    args = parser.parse_args()
    main(force=args.force, keep_going=args.keep_going, style=args.style, lora=args.lora, steps=args.steps, guidance=args.guidance)
