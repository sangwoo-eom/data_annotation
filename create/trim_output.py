from pathlib import Path
from collections import defaultdict
import random
import argparse


# =========================
# CLI
# =========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dir",
    type=str,
    required=True,
    help="target directory (e.g. output_1 or output_2)"
)

parser.add_argument(
    "--keep",
    type=int,
    default=200
)

args = parser.parse_args()

TARGET_DIR = Path(args.dir)
KEEP_PER_CLASS = args.keep


# =========================
# PATHS
# =========================

LABEL_DIR = TARGET_DIR / "labels"
MASK_DIR = TARGET_DIR / "masks"
OVERLAY_DIR = TARGET_DIR / "images"


# =========================
# util
# =========================

def extract_class_id(stem):

    parts = stem.split("_")

    for p in parts:
        if p.isdigit():
            return int(p)

    return None


# =========================
# MAIN
# =========================

def main():

    label_files = list(LABEL_DIR.glob("*.txt"))

    class_map = defaultdict(list)

    for f in label_files:

        stem = f.stem

        cls = extract_class_id(stem)

        if cls is None:
            continue

        class_map[cls].append(stem)

    total_deleted = 0

    for cls, stems in class_map.items():

        if len(stems) <= KEEP_PER_CLASS:
            print(f"class {cls}: {len(stems)} (skip)")
            continue

        random.shuffle(stems)

        keep = set(stems[:KEEP_PER_CLASS])
        delete = stems[KEEP_PER_CLASS:]

        print(f"class {cls}: {len(stems)} → keep {KEEP_PER_CLASS}, delete {len(delete)}")

        for stem in delete:

            label_path = LABEL_DIR / f"{stem}.txt"
            mask_path = MASK_DIR / f"{stem}.png"
            overlay_path = OVERLAY_DIR / f"{stem}.png"

            if label_path.exists():
                label_path.unlink()

            if mask_path.exists():
                mask_path.unlink()

            if overlay_path.exists():
                overlay_path.unlink()

            total_deleted += 1

    print("\nDeleted:", total_deleted)


if __name__ == "__main__":
    main()