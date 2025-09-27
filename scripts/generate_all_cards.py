import argparse
import json, os, subprocess, sys

CARDS = "assets/meta/cards.json"
PY = sys.executable or "python"


def art_path(art_id: str) -> str:
    return os.path.join("assets", "art", "cards", f"{art_id}.png")


def main(force: bool, keep_going: bool) -> None:
    with open(CARDS, "r", encoding="utf-8") as f:
        cards = json.load(f)

    made, skipped = 0, 0
    for card in cards:
        art_id = card["art_id"]
        prompt = card["art_prompt"]
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
    args = parser.parse_args()
    main(force=args.force, keep_going=args.keep_going)
