import os
import importlib
from datetime import datetime

from train.utils import get_train_model_save_dir

COLORS = ['gray', 'color']
STRATEGIES = [1, 2, 3]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 결과 파일 이름 생성 (모델 폴더 마지막 이름 기준)
today_str = datetime.now().strftime("%Y%m%d")
# 초기 시퀀스 번호 1부터 시작, 동일 이름 파일이 있으면 증가
seq = 1
while True:
    result_filename = f"{today_str}_{seq}.txt"
    RESULTS_PATH = os.path.join(RESULTS_DIR, result_filename)
    if not os.path.exists(RESULTS_PATH):
        break
    seq += 1

# 결과 파일 초기화
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write(f"테스트 결과 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
    f.write("=" * 50 + "\n\n")

for color in COLORS:
    for strategy in STRATEGIES:
        print(f"\n\n=== {color.upper()} - 전략 {strategy} ===")

        # 학습 폴더 생성
        TRAIN_MODEL_SAVE_DIR = get_train_model_save_dir(color, strategy)

        # =====================
        # 학습 수행
        # =====================
        train_mod = importlib.import_module("train.train")
        importlib.reload(train_mod)
        best_epoch = train_mod.train_model(color, strategy, TRAIN_MODEL_SAVE_DIR)
        print(f"▶ 학습 완료 (최고 검증 정확도 epoch: {best_epoch})")

        # =====================
        # 테스트 수행
        # =====================
        test_mod = importlib.import_module("test.test")
        importlib.reload(test_mod)

        model_path = os.path.join(TRAIN_MODEL_SAVE_DIR, f"epoch_{best_epoch:02d}.h5")
        results_txt_path = os.path.join(TRAIN_MODEL_SAVE_DIR, f"results_{best_epoch}.txt")
        confusion_matrix_path = os.path.join(TRAIN_MODEL_SAVE_DIR, f"confusion_matrix_{best_epoch}.png")

        # 테스트 수행
        test_mod.test_model(
            selected_color=color,
            selected_strategy=strategy,
            model_path=model_path,
            epoch=best_epoch,
            results_txt_dir=TRAIN_MODEL_SAVE_DIR,
            confusion_matrix_dir=TRAIN_MODEL_SAVE_DIR
        )

        # =====================
        # 결과 기록 (results.txt에 누적)
        # =====================
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(f"{color.upper()} - 전략 {strategy} (최고 검증 정확도 epoch: {best_epoch})\n")
            f.write("-" * 40 + "\n")
            if os.path.exists(results_txt_path):
                with open(results_txt_path, "r", encoding="utf-8") as r:
                    f.write(r.read())
            f.write("\n\n")

print(f"\n모든 테스트 완료. 결과 저장: {RESULTS_PATH}")
