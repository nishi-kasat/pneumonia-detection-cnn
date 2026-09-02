"""Grad-CAM utilities for the pneumonia CNN.

The helper assumes a binary Keras classifier with a sigmoid output. It computes
an activation heatmap for the positive (Pneumonia) class and overlays it on the
original chest X-ray.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def find_last_conv_layer(model):
    """Return the name of the last 2-D convolutional layer."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """Generate a Grad-CAM heatmap for the model's sigmoid output."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pneumonia_score = predictions[:, 0]

    grads = tape.gradient(pneumonia_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy(), float(predictions[0, 0].numpy()), last_conv_layer_name


def overlay_gradcam(image_rgb, heatmap, alpha=0.4):
    """Overlay a Grad-CAM heatmap on an RGB image."""
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * image_rgb + alpha * heatmap, 0, 255).astype(np.uint8)
