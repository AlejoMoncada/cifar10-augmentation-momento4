"""Exact Momento 3 CNN architecture, with injectable augmentation block."""

import tensorflow as tf
from tensorflow.keras import layers, models

from .augmentation import build_augmentation


def build_cnn_model(augmentation=None, input_shape=(32, 32, 3), num_classes=10):
    """Reproduce the reference CNN; only its augmentation block may differ."""
    if augmentation is None:
        augmentation = build_augmentation("reference")
    inputs = layers.Input(shape=input_shape)
    x = augmentation(inputs)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_CIFAR10_Reproducible")
