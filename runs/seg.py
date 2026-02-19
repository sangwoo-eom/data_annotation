# runs/seg.py
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pipeline.common as C

def main():
    C.set_output_root("output_seg")

    model = YOLO(str(C.BASE_DIR / "weights" / "yolo26l-seg.pt"))

    img_paths = list(C.INPUT_DIR.glob("*.jpg"))
    img_count = len(img_paths)
    assert img_count > 0, "No input images found"

    print(f"[SEG ONLY] images: {img_count}")

    t0 = time.time()

    for img_path in img_paths:
        image_id = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, verbose=False)[0]
        if results.masks is None:
            continue

        masks = results.masks.data.cpu().numpy()  # (N, Hm, Wm)

        for i, mask in enumerate(masks):
            mask = (mask * 255).astype("uint8")

            # mask 해상도 → 원본 해상도로 복원
            mask = cv2.resize(
                mask,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            name = f"{image_id}_obj{i}.png"
            cv2.imwrite(str(C.OUT_IMG / "04_seg" / name), mask)

            # overlay (float32로 블렌딩 후 uint8 복귀)
            overlay = img.copy().astype(np.float32)
            overlay[mask > 0] = (
                0.5 * overlay[mask > 0]
                + 0.5 * np.array([0, 255, 0], dtype=np.float32)
            )
            overlay = overlay.astype(np.uint8)

            cv2.imwrite(
                str(C.OUT_IMG / "04_seg" / f"{image_id}_overlay_{i}.jpg"),
                overlay
            )

    t1 = time.time()
    total_time = t1 - t0
    fps = img_count / total_time

    print(f"[SEG ONLY] total time: {total_time:.3f}s")
    print(f"[SEG ONLY] FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
