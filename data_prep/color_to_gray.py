import os
import shutil
import cv2
from data_prep.utils import OUTPUT_COLOR_DIRS, OUTPUT_GRAY_DIRS

def convert_color_to_gray():
    print("컬러 이미지 → 흑백 이미지 변환 시작...")
    total_images_processed = 0
    total_labels_copied = 0
    
    for key in OUTPUT_COLOR_DIRS:
        color_image_dir = os.path.join(OUTPUT_COLOR_DIRS[key], 'images')
        gray_image_dir  = os.path.join(OUTPUT_GRAY_DIRS[key], 'images')
        color_label_dir = os.path.join(OUTPUT_COLOR_DIRS[key], 'labels')
        gray_label_dir  = os.path.join(OUTPUT_GRAY_DIRS[key], 'labels')
        
        os.makedirs(gray_image_dir, exist_ok=True)
        os.makedirs(gray_label_dir, exist_ok=True)
        
        # 이미지 변환
        files = sorted(f for f in os.listdir(color_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
        print(f"\n[{key}] 총 {len(files)}장 이미지 처리 예정")
        for idx, file_name in enumerate(files, start=1):
            color_path = os.path.join(color_image_dir, file_name)
            gray_path  = os.path.join(gray_image_dir, file_name)
            
            img = cv2.imread(color_path)
            if img is None:
                print(f"⚠️ 이미지 읽기 실패: {color_path}")
                continue
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(gray_path, gray_img)
            total_images_processed += 1
            if idx % 10 == 0 or idx == len(files):
                print(f"  [{idx}/{len(files)}] 이미지 변환 완료: {file_name}")
        
        # 라벨 복사
        label_files = sorted(f for f in os.listdir(color_label_dir) if f.endswith('.txt'))
        print(f"[{key}] 총 {len(label_files)}장 라벨 처리 예정")
        for idx, label_file in enumerate(label_files, start=1):
            src_label = os.path.join(color_label_dir, label_file)
            dst_label = os.path.join(gray_label_dir, label_file)
            shutil.copy2(src_label, dst_label)
            total_labels_copied += 1
            if idx % 10 == 0 or idx == len(label_files):
                print(f"  [{idx}/{len(label_files)}] 라벨 복사 완료: {label_file}")
    
    print(f"\n모든 이미지 흑백 변환 및 라벨 복사 완료!")
    print(f"총 이미지 변환: {total_images_processed}장")
    print(f"총 라벨 복사: {total_labels_copied}장")

if __name__ == "__main__":
    convert_color_to_gray()
