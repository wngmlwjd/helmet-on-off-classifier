import tensorflow as tf
from tensorflow.keras import layers, models
from data_prep.utils import TARGET_SIZE

# ===============================
# model_2 에서 조금 변형 (정확도 낮추기 위해 Dropout 추가)
# ===============================

def main_cnn(selected_color):
    w, h = TARGET_SIZE
    d = 3 if selected_color == 'color' else 1

    inputs = tf.keras.Input(shape=(h, w, d))

    # 1번째 Conv + MaxPooling
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # 2번째 Conv + MaxPooling
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Dropout 추가 (정확도 낮추기)
    x = layers.Dropout(0.5)(x)

    # Flatten 후 Dense
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)
