"""Seeded, in-model CIFAR-10 augmentation scenarios."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

SCENARIOS = ("reference", "full", "flip_h", "rotation", "zoom", "flip_v", "blur")


class RandomGaussianBlur(layers.Layer):
    """Training-only depthwise Gaussian blur with a sampled sigma."""

    def __init__(self, seed: int = 42, sigma_min: float = 0.5, sigma_max: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.generator = tf.keras.random.SeedGenerator(seed)

    def call(self, inputs, training=None):
        def apply_blur():
            sigma = tf.keras.random.uniform((), self.sigma_min, self.sigma_max, seed=self.generator)
            coords = tf.range(-3, 4, dtype=tf.float32)
            kernel_1d = tf.exp(-0.5 * tf.square(coords / sigma))
            kernel_1d /= tf.reduce_sum(kernel_1d)
            kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
            kernel = tf.reshape(kernel_2d, (7, 7, 1, 1))
            kernel = tf.tile(kernel, (1, 1, tf.shape(inputs)[-1], 1))
            padded = tf.pad(inputs, ((0, 0), (3, 3), (3, 3), (0, 0)), mode="REFLECT")
            return tf.nn.depthwise_conv2d(padded, kernel, strides=[1, 1, 1, 1], padding="VALID")

        if isinstance(training, bool):
            return apply_blur() if training else inputs
        return tf.cond(tf.cast(training if training is not None else False, tf.bool), apply_blur, lambda: inputs)

    def get_config(self):
        return {**super().get_config(), "seed": self.seed, "sigma_min": self.sigma_min, "sigma_max": self.sigma_max}


def build_augmentation(scenario: str, seed: int = 42) -> tf.keras.Sequential:
    """Return a preprocessing block that is identity during inference."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    builders = {
        "flip_h": lambda: layers.RandomFlip("horizontal", seed=seed),
        "rotation": lambda: layers.RandomRotation(0.1, seed=seed),
        "zoom": lambda: layers.RandomZoom(0.1, seed=seed),
        "flip_v": lambda: layers.RandomFlip("vertical", seed=seed),
        "blur": lambda: RandomGaussianBlur(seed=seed),
    }
    components = ("flip_h", "rotation", "zoom") if scenario == "full" else (() if scenario == "reference" else (scenario,))
    transforms = [builders[name]() for name in components]
    # Keras cannot call an empty Sequential inside a Functional model.
    # Identity adds no augmentation behavior or trainable parameters.
    if not transforms:
        transforms = [layers.Identity()]
    return tf.keras.Sequential(transforms, name=f"augmentation_{scenario}")


def assert_train_only(model: tf.keras.Model, augmentation: tf.keras.Sequential, images: np.ndarray) -> None:
    """Verify deterministic model inference and exact augmentation identity."""
    x = tf.convert_to_tensor(images, dtype=tf.float32)
    first = model(x, training=False).numpy()
    second = model(x, training=False).numpy()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(augmentation(x, training=False).numpy(), x.numpy())
