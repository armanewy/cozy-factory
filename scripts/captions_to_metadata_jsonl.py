import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Convert captions.json to metadata.jsonl for datasets imagefolder")
    ap.add_argument("--root", required=True, help="dataset root containing images/ and captions.json")
    args = ap.parse_args()

    root = Path(args.root)
    captions = json.loads((root / "captions.json").read_text("utf-8"))
    meta_path = root / "metadata.jsonl"
    lines = []
    for fname, text in captions.items():
        # imagefolder expects relative paths from the data root
        lines.append(json.dumps({"file_name": f"images/{fname}", "text": text}, ensure_ascii=False))
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {meta_path} with {len(lines)} entries")


if __name__ == "__main__":
    main()

