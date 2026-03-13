from pathlib import Path

INPUT_DIR = Path(r"C:\Users\sangw\Desktop\Project\project\data_annotation_dh\create\input")

count = 0

for p in INPUT_DIR.iterdir():

    if not p.is_file():
        continue

    stem = p.stem
    suffix = p.suffix

    # class_2_8 → 2_8
    if stem.startswith("class_"):

        new_stem = stem.replace("class_", "", 1)

        new_name = new_stem + suffix
        new_path = p.with_name(new_name)

        if new_path.exists():
            print(f"SKIP (already exists): {new_name}")
            continue

        p.rename(new_path)

        print(f"{p.name}  →  {new_name}")

        count += 1

print("\nRenamed:", count)