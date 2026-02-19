# runs/detect.py
import time
from pathlib import Path
from pipeline.common import set_output_root, INPUT_DIR
from pipeline.step1_bbox import run as run_bbox

def main():
    set_output_root("output_detect")

    img_count = len(list(INPUT_DIR.glob("*.jpg")))
    assert img_count > 0, "No input images found"

    print(f"[DETECT] images: {img_count}")

    t0 = time.time()
    run_bbox(model_name="yolo26l.pt")
    t1 = time.time()

    total_time = t1 - t0
    fps = img_count / total_time

    print(f"[DETECT] total time: {total_time:.3f}s")
    print(f"[DETECT] FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
