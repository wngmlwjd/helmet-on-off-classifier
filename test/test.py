import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from test.utils import SELECTED_COLOR, IMG_SIZE, TEST_IMAGE_DIR, TEST_LABEL_DIR

def preprocess(img, selected_color):
    img = cv2.resize(img, IMG_SIZE)
    if selected_color == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def test_model(selected_color, selected_strategy, model_path, epoch,
               results_txt_dir="./", confusion_matrix_dir="./"):
    """
    selected_color  : 'gray' or 'color'
    selected_strategy: 전략 번호 (사용자 참고용)
    model_path      : 학습된 모델 경로
    epoch           : 테스트에 사용한 모델 epoch 번호
    results_txt_dir : 결과 TXT 저장 폴더
    confusion_matrix_dir : Confusion Matrix 이미지 저장 폴더
    """
    # 전역변수 설정
    global SELECTED_COLOR
    SELECTED_COLOR = selected_color

    os.makedirs(results_txt_dir, exist_ok=True)
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    # 파일 이름에 epoch 정보 추가
    results_txt_path = os.path.join(results_txt_dir, f"results_{epoch}.txt")
    confusion_matrix_path = os.path.join(confusion_matrix_dir, f"confusion_matrix_{epoch}.png")

    # 모델 로드
    model = tf.keras.models.load_model(model_path)

    # 테스트 이미지 리스트
    img_ext = (".jpg", ".jpeg", ".png", ".bmp")
    file_list = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(img_ext)]

    y_true, y_pred = [], []

    for file_name in file_list:
        img_path = os.path.join(TEST_IMAGE_DIR, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        label_path = os.path.join(TEST_LABEL_DIR, file_name.replace("image", "label").rsplit(".",1)[0]+".txt")
        if not os.path.exists(label_path):
            continue
        with open(label_path,"r") as f:
            gt_label = int(f.readline().strip().split()[0])

        pred_prob = model.predict(preprocess(img, selected_color), verbose=0)[0][0]
        pred_label = int(pred_prob > 0.5)

        y_true.append(gt_label)
        y_pred.append(pred_label)

    # 성능 평가
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)

    results_str = (
        f"=== 성능 평가 결과 (Epoch {epoch}) ===\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1-score : {f1:.4f}\n"
    )

    # TXT 저장
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(results_str)

    # Confusion Matrix 이미지 저장
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks([0,1], ["No Helmet","Helmet"])
    plt.yticks([0,1], ["No Helmet","Helmet"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()

    print(results_str)
    print(f"✅ 결과 TXT 저장: {results_txt_path}")
    print(f"✅ Confusion Matrix 이미지 저장: {confusion_matrix_path}")
