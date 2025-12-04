# train, test 분리 저장
import os
import shutil
import random
import time  # time.sleep 사용 중이므로 추가

from data_prep.utils import OUTPUT_COLOR_DIRS, OUTPUT_GRAY_DIRS

# -----------------------------
# 설정
# -----------------------------
BASE_DIR = "./dataset/preprocessed/gray"         # 원본 데이터셋 폴더
OUTPUT_DIR = "./dataset/preprocessed/gray"        # train/test 저장 위치

DATASET_TYPES = [
    "1. forced_scale",
    "2. padded_scale",
    "3. aspect_aware_crop"
]

train_ratio = 0.8

# -----------------------------
# 유틸 함수
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def print_progress(prefix, current, total, filename):
    percent = (current / total) * 100
    print(f"{prefix} [{current}/{total}] {percent:6.2f}% - {filename}", end="\r")


# ★★★ 추가된 부분: 이미지 파일명 → 라벨 파일명 변환 함수 ★★★
def get_label_name(image_filename):
    # image_1_crop1_forced.jpg → 1_crop1_forced 뽑기
    core = image_filename[len("image_") : image_filename.rfind(".")]
    return f"label_{core}.txt"


def split_dataset(img_dir, lbl_dir, train_img, train_lbl, test_img, test_lbl):
    # 이미지 목록 정렬
    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    total_files = len(images)
    train_count = int(total_files * train_ratio)

    train_files = images[:train_count]
    test_files = images[train_count:]

    # -----------------------------
    # Train 파일 복사
    # -----------------------------
    print(f"\n  → Train 세트 복사 시작 ({len(train_files)}개)")
    for idx, fname in enumerate(train_files, 1):
        shutil.copy(os.path.join(img_dir, fname), os.path.join(train_img, fname))

        # ★ 변경: 라벨 이름 생성 방식
        label = get_label_name(fname)
        shutil.copy(os.path.join(lbl_dir, label), os.path.join(train_lbl, label))

        print_progress("    진행중", idx, len(train_files), fname)
        time.sleep(0.001)

    print()

    # -----------------------------
    # Test 파일 복사
    # -----------------------------
    print(f"  → Test 세트 복사 시작 ({len(test_files)}개)")
    for idx, fname in enumerate(test_files, 1):
        shutil.copy(os.path.join(img_dir, fname), os.path.join(test_img, fname))

        # ★ 변경: 라벨 이름 생성 방식
        label = get_label_name(fname)
        shutil.copy(os.path.join(lbl_dir, label), os.path.join(test_lbl, label))

        print_progress("    진행중", idx, len(test_files), fname)
        time.sleep(0.001)

    print("\n")

    return len(train_files), len(test_files)

# -----------------------------
# 메인
# -----------------------------
def main():
    print("데이터 분할 작업 시작...\n")

    for ds in DATASET_TYPES:
        print(f"[{ds}] 처리 시작")

        original_dir = os.path.join(BASE_DIR, ds)
        img_dir = os.path.join(original_dir, "images")
        lbl_dir = os.path.join(original_dir, "labels")

        # 저장 경로 생성
        train_img = os.path.join(OUTPUT_DIR, "train", ds, "images")
        train_lbl = os.path.join(OUTPUT_DIR, "train", ds, "labels")
        test_img = os.path.join(OUTPUT_DIR, "test", ds, "images")
        test_lbl = os.path.join(OUTPUT_DIR, "test", ds, "labels")

        ensure_dir(train_img)
        ensure_dir(train_lbl)
        ensure_dir(test_img)
        ensure_dir(test_lbl)

        train_n, test_n = split_dataset(
            img_dir, lbl_dir,
            train_img, train_lbl,
            test_img, test_lbl
        )

        print(f" → Train: {train_n}개, Test: {test_n}개 완료\n")

    print("\n모든 데이터셋 train/test 분류 완료!")

if __name__ == "__main__":
    main()
