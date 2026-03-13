# pipeline/common.py

from pathlib import Path
import time

# ======================
# Base
# ======================
BASE_DIR = Path(__file__).resolve().parent.parent

# ======================
# Input
# ======================
IMAGE_DIR = BASE_DIR / "images"
INPUT_DIR = IMAGE_DIR / "original"

if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

# ======================
# YOLO Seg Output
# ======================
YOLO_ROOT = IMAGE_DIR / "yolo_seg_results"
YOLO_MASK_DIR = YOLO_ROOT / "masks"
YOLO_LABEL_DIR = YOLO_ROOT / "labels"

YOLO_MASK_DIR.mkdir(parents=True, exist_ok=True)
YOLO_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# Benchmark Output
# ======================
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FPS = OUTPUT_DIR / "benchmark"
OUT_FPS.mkdir(parents=True, exist_ok=True)

# ======================
# Timer
# ======================
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start