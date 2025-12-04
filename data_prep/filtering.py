# filtering.py
"""
이미지/라벨 필터링 스크립트
- 이미지 크기가 아니라, 라벨 파일의 bbox 픽셀 크기를 기준으로 필터링
- bbox 크기가 일정 기준 이상이면 이미지/라벨 유지
- 작은 bbox 줄은 제거
- bbox가 한번도 기준을 통과하지 못하면 이미지/라벨 모두 제외
"""

import os
import shutil
import cv2

# utils.py 에서 필요한 것들 import
from data_prep.utils import (
    FILTER_SIZE,
    RAW_IMAGES_DIR,
    RAW_LABELS_DIR,
    FILTERED_IMAGES_DIR,
    FILTERED_LABELS_DIR,
    get_bbox_pixel_coords
)

# bbox 최소 크기 기준 (픽셀 기준)
MIN_BBOX_WIDTH, MIN_BBOX_HEIGHT = FILTER_SIZE


def ensure_dirs():
    os.makedirs(FILTERED_IMAGES_DIR, exist_ok=True)
    os.makedirs(FILTERED_LABELS_DIR, exist_ok=True)


def get_number_from_filename(name: str) -> str:
    """파일명에서 숫자만 추출 (image_12.jpg → '12')"""
    return ''.join(ch for ch in name if ch.isdigit())


def filter_dataset():
    ensure_dirs()

    img_files = sorted(
        f for f in os.listdir(RAW_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )

    total_images = len(img_files)
    kept_images = 0
    kept_labels = 0

    print(f"원본 이미지 개수: {total_images}")

    for img_name in img_files:
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)

        # 이미지 읽기
        img = cv2.imread(raw_img_path)
        if img is None:
            print(f"  ⚠️ 이미지 읽기 실패: {raw_img_path}")
            continue

        h, w = img.shape[:2]

        # ----- 라벨 매칭 -----
        num = get_number_from_filename(img_name)
        label_name = f"label_{num}.txt"
        raw_label_path = os.path.join(RAW_LABELS_DIR, label_name)

        if not os.path.isfile(raw_label_path):
            print(f"  ⚠️ 라벨 없음 → 제외: {label_name}")
            continue

        # 라벨 읽기
        with open(raw_label_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        kept_lines = []
        for line in lines:
            # YOLO → 픽셀 변환
            cls, x_min, y_min, x_max, y_max = get_bbox_pixel_coords(line, w, h)

            if cls is None:
                print(f"    ⚠️ 라벨 포맷 오류: {line}")
                continue

            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            # bbox 크기 필터
            if bbox_w < MIN_BBOX_WIDTH or bbox_h < MIN_BBOX_HEIGHT:
                print(f"    ✂️ 작은 bbox 제외: {line}")
                continue

            kept_lines.append(line)

        # 조건을 만족하는 bbox가 하나도 없으면 이미지/라벨 제외
        if not kept_lines:
            print(f"  ✂️ 이미지 제외 (유효 bbox 없음): {img_name}")
            continue

        # 이미지 복사
        dst_img_path = os.path.join(FILTERED_IMAGES_DIR, img_name)
        shutil.copy2(raw_img_path, dst_img_path)
        kept_images += 1
        print(f"  ✅ 이미지 복사: {dst_img_path}")

        # 라벨 생성
        dst_label_path = os.path.join(FILTERED_LABELS_DIR, label_name)
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept_lines) + "\n")

        kept_labels += 1
        print(f"    ✅ 라벨 저장: {dst_label_path} (lines: {len(kept_lines)})")

    # 요약 출력
    print("\n🎉 필터링 완료")
    print(f"  - 최종 유지된 이미지: {kept_images} / {total_images}")
    print(f"  - 생성된 라벨:       {kept_labels}")


if __name__ == "__main__":
    filter_dataset()
