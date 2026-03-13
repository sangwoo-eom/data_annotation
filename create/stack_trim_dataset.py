from pathlib import Path
from collections import defaultdict
import random


# =========================
# USER SETTING
# =========================

KEEP_PER_BG = {
    "paper": 17,
    "floor": 17,
    "desk": 16
}


# =========================
# PATHS
# =========================

BASE = Path(r"C:\Users\sangw\Desktop\Project\project\data_annotation_dh\create")

DATASET = BASE / "dataset"

IMG_DIR = DATASET / "images"
LAB_DIR = DATASET / "labels"
MSK_DIR = DATASET / "masks"


# =========================
# MAIN
# =========================

def main():

    img_files = list(IMG_DIR.glob("*.png"))

    groups = defaultdict(list)

    for img in img_files:

        parts = img.stem.split("_")

        if len(parts) < 3:
            continue

        bg = parts[0]
        cls = int(parts[1])

        groups[(bg, cls)].append(img.stem)


    total_deleted = 0


    for (bg, cls), stems in groups.items():

        keep_n = KEEP_PER_BG.get(bg)

        if keep_n is None:
            continue

        if len(stems) <= keep_n:

            print(f"{bg} class{cls}: {len(stems)} (skip)")
            continue


        random.shuffle(stems)

        keep = set(stems[:keep_n])
        delete = stems[keep_n:]


        print(f"{bg} class{cls}: {len(stems)} → keep {keep_n}, delete {len(delete)}")


        for stem in delete:

            img = IMG_DIR / f"{stem}.png"
            lab = LAB_DIR / f"{stem}.txt"
            msk = MSK_DIR / f"{stem}.png"


            if img.exists():
                img.unlink()

            if lab.exists():
                lab.unlink()

            if msk.exists():
                msk.unlink()


            total_deleted += 1


    print("\nDeleted:", total_deleted)


if __name__ == "__main__":
    main()