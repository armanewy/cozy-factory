import argparse
import json, os, subprocess, sys

CARDS = "assets/meta/cards.json"
PY = sys.executable or "python"

def art_path(art_id: str) -> str:
    return os.path.join("assets", "art", "cards", f"{art_id}.png")

def main(force: bool, keep_going: bool):
    with open(CARDS, "r", encoding="utf-8") as f:
        cards = json.load(f)

    made, skipped = 0, 0
    for c in cards:
        art_id = c["art_id"]
        prompt = c["art_prompt"]
        negative = c.get("negative", "")
        out = art_path(art_id)
        if (not force) and os.path.exists(out):
            print(f"⏭️  skip {art_id} (exists)")
            skipped += 1
            continue
        # Call the single-source CLI to avoid code drift
        cmd = [
            PY, "scripts/generate_card.py",
            "--subject", prompt,
            "--id", art_id,
            "--negative", negative
        ]
        print("▶", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
            made += 1
        except subprocess.CalledProcessError as e:
            if keep_going:
                print(f"❌ {art_id} failed ({e.returncode}), continuing...")
                continue
            raise
        made += 1

    print(f"✅ done. generated={made} skipped={skipped}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--continue", dest="keep_going", action="store_true",
                    help="continue on individual card errors")
    args = ap.parse_args()
    main(force=args.force, keep_going=args.keep_going)