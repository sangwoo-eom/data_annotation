# Data Annotation Pipeline

YOLO-based detection + SAM segmentation pipeline.

## Setup

pip install -r requirements.txt

## Weights

Place the following files inside the `weights/` directory:

- yolo26l.pt
- yolo26l-seg.pt
- sam_vit_b_01ec64.pth

## Run

Detection only:
python runs/detect.py

Detection + SAM segmentation:
python runs/detect_seg.py

Segmentation only (YOLO-seg):
python runs/seg.py
