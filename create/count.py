from pathlib import Path
from collections import defaultdict

# =========================
# 경로
# =========================

DIR = Path(r"C:\Users\sangw\Desktop\Project\project\data_annotation_dh\create\dataset\images")

# =========================
# count
# =========================

counts = defaultdict(int)

for img_path in DIR.glob("*.png"):

    stem = img_path.stem

    parts = stem.split("_")

    cls = None

    # 숫자인 부분 찾기
    for p in parts:
        if p.isdigit():
            cls = int(p)
            break

    if cls is None:
        continue

    counts[cls] += 1


# =========================
# 출력
# =========================

total = 0

print("\nClass counts\n")

for cls in sorted(counts.keys()):
    n = counts[cls]
    print(f"class {cls}: {n}")
    total += n

print("\nTotal:", total)