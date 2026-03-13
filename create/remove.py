from pathlib import Path
import shutil
import argparse

BASE = Path.cwd()

TARGETS = {
    "dataset": [
        BASE / "create/dataset/images",
        BASE / "create/dataset/labels",
        BASE / "create/dataset/masks",
    ],
    "input": [
        BASE / "create/input",
    ],
    "output1": [
        BASE / "create/output_1/labels",
        BASE / "create/output_1/masks",
        BASE / "create/output_1/images",
    ],
    "output2": [
        BASE / "create/output_2/images",
        BASE / "create/output_2/labels",
        BASE / "create/output_2/masks",
    ],
    "training": [
        BASE / "create/training/images/train",
        BASE / "create/training/images/val",
        BASE / "create/training/labels/train",
        BASE / "create/training/labels/val",
    ],
}

TARGETS["all"] = sum(TARGETS.values(), [])


def clear_folder(folder):
    if not folder.exists():
        print("[skip]", folder)
        return

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    print("[clean]", folder)


def main(mode):
    if mode not in TARGETS:
        print("unknown mode:", mode)
        return

    print("BASE =", BASE)

    for folder in TARGETS[mode]:
        clear_folder(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dataset")

    args = parser.parse_args()

    main(args.mode)