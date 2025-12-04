# 종횡비 + 너비/높이 평균 계산 스크립트 (픽셀 단위 버전)
import os
import cv2
from data_prep.utils import FILTERED_LABELS_DIR, FILTERED_IMAGES_DIR, get_bbox_pixel_coords

def compute_bbox_statistics():
    label_files = [
        f for f in os.listdir(FILTERED_LABELS_DIR)
        if f.lower().endswith('.txt')
    ]

    print(f"📄 총 라벨 파일 개수: {len(label_files)}")

    widths = []
    heights = []
    ratios = []

    for label_file in label_files:
        label_path = os.path.join(FILTERED_LABELS_DIR, label_file)

        # 이미지 파일명 매칭
        img_name = label_file.replace("label_", "image_").replace(".txt", ".jpg")
        img_path = os.path.join(FILTERED_IMAGES_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"⚠️ 이미지 없음: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 이미지 읽기 실패: {img_path}")
            continue

        h, w = img.shape[:2]

        # 라벨 파일 읽기
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for line in lines:
            cls, x_min, y_min, x_max, y_max = get_bbox_pixel_coords(line, w, h)

            if cls is None:
                continue

            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            if bbox_w <= 0 or bbox_h <= 0:
                continue

            widths.append(bbox_w)
            heights.append(bbox_h)
            ratios.append(bbox_w / bbox_h)

    # ===== 출력 =====
    if widths:
        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)
        avg_ratio = sum(ratios) / len(ratios)

        print("\n📊 ==== BBOX 픽셀 단위 통계 결과 ====")
        print(f"📌 총 bbox 개수         : {len(widths)}")
        print(f"📏 평균 너비(px)         : {avg_w:.2f}")
        print(f"📐 평균 높이(px)         : {avg_h:.2f}")
        print(f"📦 평균 종횡비(w/h)      : {avg_ratio:.4f}")
    else:
        print("⚠️ 변환 가능한 bbox가 없습니다.")

if __name__ == "__main__":
    compute_bbox_statistics()
