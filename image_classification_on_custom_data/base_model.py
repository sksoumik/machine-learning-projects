import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def create_model(num_classes, data_aug):
    """
    It creates a model that takes in images, rescales them, applies a few convolutional layers, max
    pooling layers, dropout, flattens the result, and then applies a few dense layers
    :param num_classes: The number of classes in the dataset
    :param data_aug: This is the data augmentation layer we created earlier
    :return: ML model
    """
    model = Sequential(
        [
            data_aug,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes),
        ]
    )
    return model
