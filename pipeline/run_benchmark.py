# pipeline/run_benchmark.py

import json
from pipeline.step3_seg_base import run as step_seg
from pipeline.common import OUT_FPS

YOLO_MODELS = {
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
}

IMG_SIZES = {
    "640x640": (640, 640),
    "640x480": (640, 480),
    "768x768": (768, 768),
    "832x832": (832, 832),
    "960x720": (960, 736),
}


def summarize(times):
    avg = sum(times) / len(times) if times else 0.0
    return {
        "avg_time": avg,
        "fps": (1.0 / avg) if avg > 0 else 0.0
    }


if __name__ == "__main__":

    print("\n[WARMUP] Running dummy warm-up...")
    warmup_weight = "yolo11m.pt"
    first_imgsz = next(iter(IMG_SIZES.values()))
    _ = step_seg(weight=warmup_weight, imgsz=first_imgsz, save_output=False)
    print("[WARMUP] Done.\n")

    # 🔥 전체 결과 저장용 dict
    final_results = {
        model: {} for model in YOLO_MODELS.keys()
    }

    for size_name, imgsz in IMG_SIZES.items():

        for model_name, weight in YOLO_MODELS.items():

            seg_times = step_seg(
                weight=weight,
                imgsz=imgsz,
                save_output=True
            )

            report = summarize(seg_times)
            final_results[model_name][size_name] = report["fps"]

            # JSON 개별 저장
            out_path = OUT_FPS / f"{model_name}_{size_name}.json"
            with open(out_path, "w") as f:
                json.dump({
                    "model": model_name,
                    "imgsz": size_name,
                    "fps": report["fps"],
                    "avg_time": report["avg_time"]
                }, f, indent=2)

    # ================= FINAL SUMMARY =================
    print("\n================ FINAL BENCHMARK SUMMARY ================\n")

    header = "Model".ljust(10)
    for size_name in IMG_SIZES.keys():
        header += f"| {size_name:8s} "
    print(header)
    print("-" * len(header))

    for model_name in YOLO_MODELS.keys():
        row = model_name.ljust(10)
        for size_name in IMG_SIZES.keys():
            fps = final_results[model_name].get(size_name, 0.0)
            row += f"| {fps:8.2f} "
        print(row)

    print("\n=========================================================\n")