import argparse
import json
import os
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Verify caption coverage for a cloth image folder")
    p.add_argument("--cloth_dir",     required=True, help="Path to folder of cloth images")
    p.add_argument("--captions_json", required=True, help="Path to captions.json")
    p.add_argument("--examples",      type=int, default=5, help="Number of random examples to print")
    return p.parse_args()


def main():
    args = parse_args()

    cloth_dir = Path(args.cloth_dir)
    if not cloth_dir.is_dir():
        raise SystemExit(f"ERROR: --cloth_dir does not exist: {cloth_dir}")

    if not os.path.exists(args.captions_json):
        raise SystemExit(f"ERROR: --captions_json does not exist: {args.captions_json}")

    image_names = {
        p.name for p in cloth_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    }

    with open(args.captions_json, "r") as f:
        captions: dict = json.load(f)

    captioned = image_names & captions.keys()
    missing   = image_names - captions.keys()
    extra     = captions.keys() - image_names   # captions with no matching image

    failed_log = Path(args.captions_json).with_name("failed_captions.txt")
    failed = set()
    if failed_log.exists():
        with open(failed_log) as f:
            failed = {line.strip() for line in f if line.strip()}

    print("=" * 60)
    print("  Caption verification report")
    print("=" * 60)
    print(f"  Total images in cloth_dir : {len(image_names)}")
    print(f"  Captioned                 : {len(captioned)}")
    print(f"  Missing captions          : {len(missing)}")
    print(f"  Logged as failed          : {len(failed)}")
    print(f"  Extra (caption, no image) : {len(extra)}")
    print("=" * 60)

    if missing:
        sample_missing = sorted(missing)[:10]
        print(f"\nFirst up to 10 missing images:")
        for name in sample_missing:
            status = " [in failed log]" if name in failed else ""
            print(f"  {name}{status}")

    n = min(args.examples, len(captioned))
    if n > 0:
        print(f"\n{n} random caption examples:")
        print("-" * 60)
        for name in random.sample(sorted(captioned), n):
            print(f"  {name}")
            print(f"    → {captions[name]}")
        print("-" * 60)

    coverage = len(captioned) / len(image_names) * 100 if image_names else 0
    print(f"\nCoverage: {coverage:.1f}%")

    if missing:
        uncovered = missing - failed
        if uncovered:
            print(f"  {len(uncovered)} images have no caption and are NOT in the failed log.")
            print("  Re-run generate_captions.py with --resume to fill them in.")


if __name__ == "__main__":
    main()
