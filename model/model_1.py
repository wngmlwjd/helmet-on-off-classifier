import tensorflow as tf
from tensorflow.keras import layers, models
from data_prep.utils import TARGET_SIZE

def main_cnn(selected_color):
    """
    3-Conv Mini-CNN
    selected_color: 'gray' 또는 'color' → 입력 채널 결정
    반환: Keras Model 객체
    """
    w, h = TARGET_SIZE
    d = 3 if selected_color == 'color' else 1

    inputs = tf.keras.Input(shape=(h, w, d))

    # 1번째 Conv + MaxPooling
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)

    # 2번째 Conv + MaxPooling
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    # 3번째 Conv + MaxPooling
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    # Global Average Pooling 후 Dense
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)
