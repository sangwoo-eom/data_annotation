# common.py
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "input/rgb"

CACHE_DIR  = BASE_DIR / "cache"
CACHE_REC  = CACHE_DIR / "records"
CACHE_CROP = CACHE_DIR / "crops"
CACHE_MASK = CACHE_DIR / "masks"

# ⭐ 런타임에 바뀔 변수
OUT_ROOT = None
OUT_IMG  = None
OUT_META = None

def set_output_root(name: str):
    global OUT_ROOT, OUT_IMG, OUT_META

    OUT_ROOT = BASE_DIR / "output" / name
    OUT_IMG  = OUT_ROOT / "images"
    OUT_META = OUT_ROOT / "meta"

    # 표준화된 디렉토리
    for d in [
        OUT_IMG / "01_input",
        OUT_IMG / "02_bbox",
        OUT_IMG / "03_crop",
        OUT_IMG / "04_seg",
        OUT_META / "bbox",
        OUT_META / "seg",
    ]:
        d.mkdir(parents=True, exist_ok=True)

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start