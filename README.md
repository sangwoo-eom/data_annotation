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


# 사용 명령어

- 프레임 추출 - python create/bag_to_images.py --start 1 --end 7 --count 300
- 디렉토리 비우기 - python create/remove.py --mode all
- 파일 이름 수정[class_2_8 → 2_8] - python create/fix_name.py
- 1차 데이터 생성[이름이 반드시 0_0 형식] - python create/generate_dataset.py
- 수정[GUI] - python create/GUI.py
- Dataset으로 이동 - python create/export_dataset.py --mode copy --bg paper
- 학습 데이터 생성 - python create/build_dataset.py --num-classes 7 --val-ratio 0.2
- 클래스별 이미지 확인 - python create/count.py
- 파일 개수 맞추기 - python create/trim_output.py --dir create/dataset --keep 67
- 학습 명령어 - yolo segment train data=dataset.yaml model=/home/prml513/project/data_annotation_dh/weights/yolo26l-seg.pt epochs=150 imgsz=640 batch=32 device=0 workers=8 close_mosaic=10 cache=True
- 데이터 누적 시 - python create/stack_dataset.py --bg desk
