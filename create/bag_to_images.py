import pyrealsense2 as rs
import cv2
import numpy as np
import argparse
from pathlib import Path


# =========================
# PATHS
# =========================

BASE_DIR = Path(r"C:\Users\sangw\Desktop\Project\project\data_annotation_dh")

BAG_DIR = BASE_DIR / "bag_paper"

OUTPUT_DIR = BASE_DIR / "create" / "input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CLI
# =========================

parser = argparse.ArgumentParser()

parser.add_argument("--start", type=int, required=True)
parser.add_argument("--end", type=int, required=True)

parser.add_argument(
    "--count",
    type=int,
    required=True,
    help="number of images to extract from each bag"
)

args = parser.parse_args()

START = args.start
END = args.end
TARGET_COUNT = args.count


# =========================
# BAG PROCESS
# =========================

def process_bag(bag_path):

    class_id = bag_path.stem

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    config.enable_stream(rs.stream.color)

    pipeline.start(config)

    playback = pipeline.get_active_profile().get_device().as_playback()
    playback.set_real_time(False)

    frames_all = []

    print("\nReading bag:", bag_path)

    while True:

        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError:
            break

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        frames_all.append(image)

    pipeline.stop()

    total_frames = len(frames_all)

    print("Total frames:", total_frames)

    if total_frames == 0:
        print("No frames found.")
        return

    if TARGET_COUNT >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(
            0,
            total_frames - 1,
            TARGET_COUNT,
            dtype=int
        )

    for i, frame_idx in enumerate(indices, start=1):

        image = frames_all[frame_idx]

        filename = f"{class_id}_{i}.png"
        save_path = OUTPUT_DIR / filename

        cv2.imwrite(str(save_path), image)

    print("Saved:", len(indices))


# =========================
# MAIN
# =========================

def main():

    for i in range(START, END + 1):

        bag_path = BAG_DIR / f"{i}.bag"

        if not bag_path.exists():
            print("Skip (not found):", bag_path)
            continue

        process_bag(bag_path)


if __name__ == "__main__":
    main()