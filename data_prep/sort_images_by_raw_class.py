import os
import shutil
from data_prep.utils import FILTERED_IMAGES_DIR, FILTERED_LABELS_DIR

def sort_images_by_filtered_class(output_root="sorted_images_filtered"):
    """라벨 파일의 클래스 값(FILTERED_LABELS_DIR 기준)에 따라 이미지를 각 폴더로 분류합니다 (파일 이름 그대로 유지)."""
    
    if not os.path.exists(FILTERED_LABELS_DIR):
        print(f"❌ 라벨 디렉터리 '{FILTERED_LABELS_DIR}'가 없습니다.")
        return
    if not os.path.exists(FILTERED_IMAGES_DIR):
        print(f"❌ 이미지 디렉터리 '{FILTERED_IMAGES_DIR}'가 없습니다.")
        return

    os.makedirs(output_root, exist_ok=True)

    label_files = sorted(f for f in os.listdir(FILTERED_LABELS_DIR) if f.endswith(".txt"))
    if not label_files:
        print(f"⚠️ '{FILTERED_LABELS_DIR}' 폴더에 라벨 파일이 없습니다.")
        return

    for label_file in label_files:
        label_path = os.path.join(FILTERED_LABELS_DIR, label_file)
        base_name = os.path.splitext(label_file)[0]
        image_name = base_name.replace("label_", "image_") + ".jpg"
        image_path = os.path.join(FILTERED_IMAGES_DIR, image_name)

        if not os.path.exists(image_path):
            print(f"⚠️ {image_name} 파일이 존재하지 않아 건너뜁니다.")
            continue

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if not first_line:
                    print(f"⚠️ {label_file} 라벨이 비어 있음. 건너뜀.")
                    continue
                cls = int(first_line.split()[0])  # 첫 번째 값이 클래스
        except Exception as e:
            print(f"⚠️ {label_file} 읽기 오류: {e}")
            continue

        # --- 클래스별 폴더 생성 ---
        class_dir = os.path.join(output_root, f"class_{cls}")
        os.makedirs(class_dir, exist_ok=True)

        # --- 이미지 복사 (이름 유지) ---
        shutil.copy2(image_path, os.path.join(class_dir, image_name))

        print(f"✅ {image_name} → class_{cls}/ 로 복사 완료")

    print("\n🎯 모든 이미지가 filtered 라벨 기준으로 클래스별로 분류되었습니다!")

if __name__ == "__main__":
    sort_images_by_filtered_class()
