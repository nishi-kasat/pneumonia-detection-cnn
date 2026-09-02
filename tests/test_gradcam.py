import numpy as np
import tensorflow as tf

from explainability.gradcam import (
    find_last_conv_layer,
    make_gradcam_heatmap,
    overlay_gradcam,
)


def build_tiny_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", name="conv_a")(inputs)
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", name="conv_b")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def test_find_last_conv_layer():
    model = build_tiny_model()
    assert find_last_conv_layer(model) == "conv_b"


def test_gradcam_shape_and_score():
    model = build_tiny_model()
    batch = np.random.default_rng(42).random((1, 32, 32, 3), dtype=np.float32)

    heatmap, score, layer_name = make_gradcam_heatmap(batch, model)

    assert layer_name == "conv_b"
    assert heatmap.ndim == 2
    assert heatmap.shape == (28, 28)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0
    assert 0.0 <= score <= 1.0


def test_overlay_preserves_image_shape():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    heatmap = np.ones((28, 28), dtype=np.float32)

    overlay = overlay_gradcam(image, heatmap)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
