# runs/detect_seg.py
import time
from pathlib import Path
from pipeline.common import set_output_root, INPUT_DIR
from pipeline.step1_bbox import run as run_bbox
from pipeline.step2_crop import run as run_crop
from pipeline.step3_seg import run as run_seg

def main():
    set_output_root("output_detect_seg")

    img_count = len(list(INPUT_DIR.glob("*.jpg")))
    assert img_count > 0, "No input images found"

    print(f"[DETECT+SEG] images: {img_count}")

    t0 = time.time()
    run_bbox(model_name="yolo26l.pt")
    run_crop()
    run_seg()
    t1 = time.time()

    total_time = t1 - t0
    fps = img_count / total_time

    print(f"[DETECT+SEG] total time: {total_time:.3f}s")
    print(f"[DETECT+SEG] FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
