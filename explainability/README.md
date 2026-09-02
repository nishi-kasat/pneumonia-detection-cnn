# Explainable Pneumonia Detection

This extension adds **Grad-CAM (Gradient-weighted Class Activation Mapping)** to the existing CNN pneumonia classifier.

## What it adds

- Identifies the final convolutional layer automatically.
- Computes a Grad-CAM heatmap for the model's pneumonia score.
- Overlays the heatmap on a chest X-ray so the influential image regions are visible.
- Keeps the original CNN architecture and classification workflow unchanged.

## Notebook integration

After training and saving the existing `model`, run:

```python
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from explainability.gradcam import make_gradcam_heatmap, overlay_gradcam

img_path = "/content/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_rgb = image.img_to_array(img).astype("uint8")
img_array = np.expand_dims(img_rgb.astype("float32") / 255.0, axis=0)

heatmap, pneumonia_score, layer_name = make_gradcam_heatmap(img_array, model)
cam = overlay_gradcam(img_rgb, heatmap)

label = "PNEUMONIA" if pneumonia_score >= 0.5 else "NORMAL"
print(f"Prediction: {label}")
print(f"Pneumonia probability: {pneumonia_score:.3f}")
print(f"Grad-CAM layer: {layer_name}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cam)
plt.title(f"Explanation: {label}")
plt.axis("off")
plt.tight_layout()
plt.show()
```

## Interpretation

Grad-CAM highlights regions that contributed to the model's pneumonia prediction. It is an **interpretability aid**, not a clinical diagnostic tool. A highlighted region should not be interpreted as proof that pneumonia is present, and model explanations should be evaluated alongside model performance and clinical expertise.

## Next XAI extensions

A useful future experiment is to compare Grad-CAM with LIME or SHAP on the same test images and study whether the explanations are stable across correctly and incorrectly classified cases.
