# pipeline/step3_seg.py

import cv2
import torch
import numpy as np
from ultralytics import YOLO
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

    # 🔥 전체 누적 통계
    stat = {
        "load": 0.0,
        "infer": 0.0,
        "cpu_copy": 0.0,
        "resize": 0.0,
        "save_png": 0.0,
        "contour": 0.0,
        "label_write": 0.0,
        "total": 0.0
    }

    for img_path in img_paths:

        with timer() as total_timer:

            # 1️⃣ 이미지 로드
            with timer() as t:
                img = cv2.imread(str(img_path))
            stat["load"] += t()

            if img is None:
                continue

            # 2️⃣ 추론
            with timer() as t:
                results = model(
                    img,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                    retina_masks=True,
                    conf=0.25
                )[0]
            stat["infer"] += t()

            if results.masks is None:
                stat["total"] += total_timer()
                times.append(total_timer())
                continue

            # 3️⃣ GPU → CPU
            with timer() as t:
                masks = results.masks.data.cpu().numpy()
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
            stat["cpu_copy"] += t()

            image_id = img_path.stem

            if save_output:
                label_path = C.YOLO_LABEL_DIR / f"{image_id}_detected.txt"
                lf = open(label_path, "w")

            for i, mask in enumerate(masks):

                # 4️⃣ resize
                with timer() as t:
                    mask_bin = (mask > 0.5).astype("uint8") * 255
                    mask_bin = cv2.resize(
                        mask_bin,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                stat["resize"] += t()

                instance_id = f"{image_id}_obj{i}"

                # 5️⃣ PNG 저장
                if save_output:
                    with timer() as t:
                        cv2.imwrite(
                            str(C.YOLO_MASK_DIR / f"{instance_id}_masked.png"),
                            mask_bin,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1]
                        )
                    stat["save_png"] += t()

                # 6️⃣ contour
                with timer() as t:
                    contours, _ = cv2.findContours(
                        mask_bin,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                stat["contour"] += t()

                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)
                poly = cnt.reshape(-1, 2)

                x1, y1, x2, y2 = boxes[i]
                cls_id = int(classes[i])

                # 7️⃣ label 저장
                if save_output:
                    with timer() as t:
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
                    stat["label_write"] += t()

            if save_output:
                lf.close()

        elapsed = total_timer()
        stat["total"] += elapsed
        times.append(elapsed)

    # 🔥 최종 통계 출력
    print("\n===== STAGE PROFILING =====")
    total = stat["total"]
    for k, v in stat.items():
        if k == "total":
            continue
        percent = (v / total) * 100 if total > 0 else 0
        print(f"{k:12s}: {v:.3f}s ({percent:5.1f}%)")

    print(f"{'TOTAL':12s}: {total:.3f}s")
    print("===========================\n")

    return times