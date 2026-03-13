# pipeline/step3_seg.py

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import pipeline.common as C
from utils.timer import timer


def run(weight, imgsz=640, save_output=True):

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(C.BASE_DIR / "weights" / weight))

    img_paths = sorted(
        list(C.INPUT_DIR.glob("*.jpg")) +
        list(C.INPUT_DIR.glob("*.png"))
    )

    if len(img_paths) == 0:
        return []

    times = []

    for img_path in img_paths:

        with timer() as get_elapsed:

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            results = model(
                img,
                imgsz=imgsz,
                device=device,
                verbose=False,
                retina_masks=True,
                conf=0.25
            )[0]

            if results.masks is None:
                times.append(get_elapsed())
                continue

            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            image_id = img_path.stem

            if save_output:
                label_path = C.YOLO_LABEL_DIR / f"{image_id}_detected.txt"
                lf = open(label_path, "w")

            for i, mask in enumerate(masks):

                mask_bin = (mask > 0.5).astype("uint8") * 255

                mask_bin = cv2.resize(
                    mask_bin,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                instance_id = f"{image_id}_obj{i}"

                if save_output:
                    mask_name = f"{instance_id}_masked.png"
                    cv2.imwrite(
                        str(C.YOLO_MASK_DIR / mask_name),
                        mask_bin,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1]  # 🔥 속도 최적화
                    )

                contours, _ = cv2.findContours(
                    mask_bin,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)
                poly = cnt.reshape(-1, 2)

                x1, y1, x2, y2 = boxes[i]
                cls_id = int(classes[i])

                if save_output:
                    poly_flat = []
                    for px, py in poly:
                        poly_flat.append(str(float(px)))
                        poly_flat.append(str(float(py)))

                    line = (
                        f"{instance_id} {cls_id} "
                        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                        + " ".join(poly_flat)
                    )
                    lf.write(line + "\n")

            if save_output:
                lf.close()

        times.append(get_elapsed())

    return times