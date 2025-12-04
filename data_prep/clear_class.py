import os
from data_prep.utils import FILTERED_LABELS_DIR

def remap_class_2_to_0():
    print("클래스 값 2 → 0 변환 시작...")
    label_files = sorted(f for f in os.listdir(FILTERED_LABELS_DIR) if f.endswith('.txt'))
    total_files = len(label_files)
    
    for idx, label_file in enumerate(label_files, start=1):
        label_path = os.path.join(FILTERED_LABELS_DIR, label_file)
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_val = parts[0]
            if cls_val == '2':
                parts[0] = '0'
            new_lines.append(" ".join(parts))
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for line in new_lines:
                f.write(line + "\n")
        
        if idx % 10 == 0 or idx == total_files:
            print(f"[{idx}/{total_files}] 라벨 변환 완료: {label_file}")
    
    print("모든 라벨 파일 클래스 값 변환 완료!")

if __name__ == "__main__":
    remap_class_2_to_0()
